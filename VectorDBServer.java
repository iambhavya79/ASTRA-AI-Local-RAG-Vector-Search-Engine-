import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class VectorDBServer {

    static final int DIMS = 16;

    // =====================================================================
    //  DATA TYPES & DISTANCE METRICS
    // =====================================================================

    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    static class Pair implements Comparable<Pair> {
        float dist;
        int id;

        Pair(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }

        @Override
        public int compareTo(Pair o) {
            int c = Float.compare(this.dist, o.dist);
            return (c != 0) ? c : Integer.compare(this.id, o.id);
        }
    }

    @FunctionalInterface
    interface DistFn {
        float apply(float[] a, float[] b);
    }

    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    }

    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m)) return VectorDBServer::cosine;
        if ("manhattan".equals(m)) return VectorDBServer::manhattan;
        return VectorDBServer::euclidean;
    }

    // =====================================================================
    //  BRUTE FORCE
    // =====================================================================

    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) {
            items.add(v);
        }

        List<Pair> knn(float[] q, int k, DistFn dist) {
            List<Pair> r = new ArrayList<>(items.size());
            for (VectorItem v : items) {
                r.add(new Pair(dist.apply(q, v.emb), v.id));
            }
            Collections.sort(r);
            if (r.size() > k) return new ArrayList<>(r.subList(0, k));
            return r;
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =====================================================================
    //  KD-TREE
    // =====================================================================

    static class KDNode {
        VectorItem item;
        KDNode left, right;

        KDNode(VectorItem v) {
            this.item = v;
        }
    }

    static class KDTree {
        KDNode root = null;
        int dims;

        KDTree(int dims) {
            this.dims = dims;
        }

        private KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
            else n.right = ins(n.right, v, d + 1);
            return n;
        }

        void insert(VectorItem v) {
            root = ins(root, v, 0);
        }

        private void knn(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<Pair> maxHeap) {
            if (n == null) return;
            float dn = dist.apply(q, n.item.emb);
            if (maxHeap.size() < k || dn < maxHeap.peek().dist) {
                maxHeap.offer(new Pair(dn, n.item.id));
                if (maxHeap.size() > k) maxHeap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;

            knn(closer, q, k, d + 1, dist, maxHeap);
            if (maxHeap.size() < k || Math.abs(diff) < maxHeap.peek().dist) {
                knn(farther, q, k, d + 1, dist, maxHeap);
            }
        }

        List<Pair> knn(float[] q, int k, DistFn dist) {
            PriorityQueue<Pair> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
            knn(root, q, k, 0, dist, maxHeap);
            List<Pair> r = new ArrayList<>(maxHeap);
            Collections.sort(r);
            return r;
        }

        void rebuild(Collection<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }

    // =====================================================================
    //  HNSW — Hierarchical Navigable Small World
    // =====================================================================

    static class HNSW {
        static class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;

            Node(VectorItem item, int maxLyr) {
                this.item = item;
                this.maxLyr = maxLyr;
                this.nbrs = new ArrayList<>(maxLyr + 1);
                for (int i = 0; i <= maxLyr; i++) this.nbrs.add(new ArrayList<>());
            }
        }

        Map<Integer, Node> G = new HashMap<>();
        int M, M0, efBuild;
        float mL;
        int topLayer = -1;
        int entryPt = -1;
        Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            this.M = m;
            this.M0 = 2 * m;
            this.efBuild = efBuild;
            this.mL = (float) (1.0 / Math.log(m));
        }

        private int randLevel() {
            float u = rng.nextFloat();
            if (u == 0.0f) u = 1e-7f;
            return (int) Math.floor(-Math.log(u) * mL);
        }

        private List<Pair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            PriorityQueue<Pair> cands = new PriorityQueue<>(); // min-heap
            PriorityQueue<Pair> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

            float d0 = dist.apply(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.offer(new Pair(d0, ep));
            found.offer(new Pair(d0, ep));

            while (!cands.isEmpty()) {
                Pair curr = cands.poll();
                if (found.size() >= ef && curr.dist > found.peek().dist) break;
                Node cNode = G.get(curr.id);
                if (cNode == null || lyr >= cNode.nbrs.size()) continue;

                for (int nid : cNode.nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);
                    float nd = dist.apply(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        cands.offer(new Pair(nd, nid));
                        found.offer(new Pair(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<Pair> res = new ArrayList<>(found);
            Collections.sort(res);
            return res;
        }

        private List<Integer> selectNbrs(List<Pair> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
                r.add(cands.get(i).id);
            }
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            G.put(id, new Node(item, lvl));

            if (entryPt == -1) {
                entryPt = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<Pair> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<Pair> W = searchLayer(item.emb, ep, efBuild, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, sel);

                for (int nid : sel) {
                    Node nNode = G.get(nid);
                    if (nNode == null) continue;
                    while (nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nNode.nbrs.get(lc);
                    conn.add(id);

                    if (conn.size() > maxM) {
                        List<Pair> ds = new ArrayList<>();
                        for (int c : conn) {
                            if (G.containsKey(c)) {
                                ds.add(new Pair(dist.apply(nNode.item.emb, G.get(c).item.emb), c));
                            }
                        }
                        Collections.sort(ds);
                        conn.clear();
                        for (int i = 0; i < Math.min(maxM, ds.size()); i++) {
                            conn.add(ds.get(i).id);
                        }
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }

            if (lvl > topLayer) {
                topLayer = lvl;
                entryPt = id;
            }
        }

        List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return Collections.emptyList();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<Pair> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            List<Pair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) return new ArrayList<>(W.subList(0, k));
            return W;
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values()) {
                for (List<Integer> layer : nd.nbrs) {
                    layer.removeIf(x -> x == id);
                }
            }
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) {
                    if (nid != id) {
                        entryPt = nid;
                        break;
                    }
                }
            }
            G.remove(id);
        }

        String getInfoJson() {
            int maxL = Math.max(topLayer + 1, 1);
            int[] nodesPerLayer = new int[maxL];
            int[] edgesPerLayer = new int[maxL];
            StringBuilder nodesJson = new StringBuilder("[");
            StringBuilder edgesJson = new StringBuilder("[");
            boolean firstNode = true, firstEdge = true;

            for (Map.Entry<Integer, Node> entry : G.entrySet()) {
                int id = entry.getKey();
                Node nd = entry.getValue();
                if (!firstNode) nodesJson.append(',');
                firstNode = false;
                nodesJson.append("{\"id\":").append(id)
                         .append(",\"metadata\":").append(jS(nd.item.metadata))
                         .append(",\"category\":").append(jS(nd.item.category))
                         .append(",\"maxLyr\":").append(nd.maxLyr).append('}');

                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    nodesPerLayer[lc]++;
                    if (lc < nd.nbrs.size()) {
                        for (int nid : nd.nbrs.get(lc)) {
                            if (id < nid) {
                                edgesPerLayer[lc]++;
                                if (!firstEdge) edgesJson.append(',');
                                firstEdge = false;
                                edgesJson.append("{\"src\":").append(id)
                                         .append(",\"dst\":").append(nid)
                                         .append(",\"lyr\":").append(lc).append('}');
                            }
                        }
                    }
                }
            }
            nodesJson.append(']');
            edgesJson.append(']');

            StringBuilder sb = new StringBuilder();
            sb.append("{\"topLayer\":").append(topLayer)
              .append(",\"nodeCount\":").append(G.size())
              .append(",\"nodesPerLayer\":").append(Arrays.toString(nodesPerLayer).replace(" ", ""))
              .append(",\"edgesPerLayer\":").append(Arrays.toString(edgesPerLayer).replace(" ", ""))
              .append(",\"nodes\":").append(nodesJson)
              .append(",\"edges\":").append(edgesJson).append('}');
            return sb.toString();
        }
    }

    // =====================================================================
    //  VECTOR DATABASE (16D Demo Index)
    // =====================================================================

    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf = new BruteForce();
        KDTree kdt;
        HNSW hnsw = new HNSW(16, 200);
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        final int dims;

        VectorDB(int d) {
            this.dims = d;
            this.kdt = new KDTree(d);
        }

        int insert(String meta, String cat, float[] emb, DistFn dist) {
            mu.lock();
            try {
                VectorItem v = new VectorItem(nextId++, meta, cat, emb);
                store.put(v.id, v);
                bf.insert(v);
                kdt.insert(v);
                hnsw.insert(v, dist);
                return v.id;
            } finally {
                mu.unlock();
            }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id);
                bf.remove(id);
                hnsw.remove(id);
                kdt.rebuild(store.values());
                return true;
            } finally {
                mu.unlock();
            }
        }

        String searchJson(float[] q, int k, String metric, String algo) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t0 = System.nanoTime();
                List<Pair> raw;
                if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
                else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
                else raw = hnsw.knn(q, k, 50, dfn);
                long us = (System.nanoTime() - t0) / 1000;

                StringBuilder ss = new StringBuilder("{\"results\":[");
                boolean first = true;
                for (Pair p : raw) {
                    VectorItem item = store.get(p.id);
                    if (item != null) {
                        if (!first) ss.append(',');
                        first = false;
                        ss.append("{\"id\":").append(item.id)
                          .append(",\"metadata\":").append(jS(item.metadata))
                          .append(",\"category\":").append(jS(item.category))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.6f", p.dist))
                          .append(",\"embedding\":").append(jVec(item.emb)).append('}');
                    }
                }
                ss.append("],\"latencyUs\":").append(us)
                  .append(",\"algo\":").append(jS(algo))
                  .append(",\"metric\":").append(jS(metric)).append('}');
                return ss.toString();
            } finally {
                mu.unlock();
            }
        }

        String benchmarkJson(float[] q, int k, String metric) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t1 = System.nanoTime(); bf.knn(q, k, dfn); long bfUs = (System.nanoTime() - t1) / 1000;
                long t2 = System.nanoTime(); kdt.knn(q, k, dfn); long kdUs = (System.nanoTime() - t2) / 1000;
                long t3 = System.nanoTime(); hnsw.knn(q, k, 50, dfn); long hnswUs = (System.nanoTime() - t3) / 1000;

                return String.format(Locale.US,
                    "{\"bruteforceUs\":%d,\"kdtreeUs\":%d,\"hnswUs\":%d,\"itemCount\":%d}",
                    bfUs, kdUs, hnswUs, store.size());
            } finally {
                mu.unlock();
            }
        }

        List<VectorItem> all() {
            mu.lock();
            try {
                return new ArrayList<>(store.values());
            } finally {
                mu.unlock();
            }
        }

        String hnswInfoJson() {
            mu.lock();
            try {
                return hnsw.getInfoJson();
            } finally {
                mu.unlock();
            }
        }

        int size() {
            mu.lock();
            try {
                return store.size();
            } finally {
                mu.unlock();
            }
        }
    }

    // =====================================================================
    //  OLLAMA CLIENT (Java 11+ HttpClient)
    // =====================================================================

    static class OllamaClient {
        String baseUrl;
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();

        OllamaClient(String host, int port) {
            this.baseUrl = "http://" + host + ":" + port;
        }

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/tags"))
                        .timeout(Duration.ofSeconds(2))
                        .GET().build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch (Exception e) {
                return false;
            }
        }

        float[] embed(String text) {
            try {
                String body = "{\"model\":" + jS(embedModel) + ",\"prompt\":" + jS(text) + "}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/embeddings"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(30))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return new float[0];
                return extractArr(res.body(), "embedding");
            } catch (Exception e) {
                return new float[0];
            }
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":" + jS(genModel) + ",\"prompt\":" + jS(prompt) + ",\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/generate"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(180))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return extractStr(res.body(), "response");
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =====================================================================
    //  DOCUMENT DATABASE (HNSW over Ollama Embeddings)
    // =====================================================================

    static class DocItem {
        int id;
        String title;
        String text;
        float[] emb;

        DocItem(int id, String title, String text, float[] emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    static class DocHit {
        float dist;
        DocItem doc;

        DocHit(float dist, DocItem doc) {
            this.dist = dist;
            this.doc = doc;
        }
    }

    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        int dims = 0;

        int insert(String title, String text, float[] emb) {
            mu.lock();
            try {
                if (dims == 0) dims = emb.length;
                DocItem item = new DocItem(nextId++, title, text, emb);
                store.put(item.id, item);
                VectorItem vi = new VectorItem(item.id, title, "doc", emb);
                hnsw.insert(vi, VectorDBServer::cosine);
                bf.insert(vi);
                return item.id;
            } finally {
                mu.unlock();
            }
        }

        List<DocHit> search(float[] q, int k, float maxDist) {
            mu.lock();
            try {
                if (store.isEmpty()) return Collections.emptyList();
                List<Pair> raw = (store.size() < 10)
                        ? bf.knn(q, k, VectorDBServer::cosine)
                        : hnsw.knn(q, k, 50, VectorDBServer::cosine);
                List<DocHit> out = new ArrayList<>();
                for (Pair p : raw) {
                    if (store.containsKey(p.id) && p.dist <= maxDist) {
                        out.add(new DocHit(p.dist, store.get(p.id)));
                    }
                }
                return out;
            } finally {
                mu.unlock();
            }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id);
                hnsw.remove(id);
                bf.remove(id);
                return true;
            } finally {
                mu.unlock();
            }
        }

        List<DocItem> all() {
            mu.lock();
            try {
                return new ArrayList<>(store.values());
            } finally {
                mu.unlock();
            }
        }

        int size() {
            mu.lock();
            try {
                return store.size();
            } finally {
                mu.unlock();
            }
        }
    }

    // =====================================================================
    //  TEXT CHUNKER & JSON/HTTP HELPERS
    // =====================================================================

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0 || (words.length == 1 && words[0].isEmpty())) return Collections.emptyList();
        if (words.length <= chunkWords) return Collections.singletonList(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }

    static String jS(String s) {
        if (s == null) return "\"\"";
        StringBuilder o = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') o.append("\\\"");
            else if (c == '\\') o.append("\\\\");
            else if (c == '\n') o.append("\\n");
            else if (c == '\r') o.append("\\r");
            else if (c == '\t') o.append("\\t");
            else o.append(c);
        }
        return o.append('"').toString();
    }

    static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(',');
            ss.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return ss.append(']').toString();
    }

    static float[] parseVec(String s) {
        if (s == null || s.trim().isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> list = new ArrayList<>();
        for (String t : parts) {
            try {
                list.add(Float.parseFloat(t.trim()));
            } catch (NumberFormatException ignored) {}
        }
        float[] res = new float[list.size()];
        for (int i = 0; i < list.size(); i++) res[i] = list.get(i);
        return res;
    }

    static String extractStr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p == -1) return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char next = body.charAt(p);
                switch (next) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(next); break;
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    static int extractInt(String body, String key, int def) {
        int p = body.indexOf("\"" + key + "\"");
        if (p == -1) return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int end = p;
        while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) end++;
        try {
            return Integer.parseInt(body.substring(p, end));
        } catch (Exception e) {
            return def;
        }
    }

    static float[] extractArr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p == -1) return new float[0];
        p = body.indexOf('[', p);
        if (p == -1) return new float[0];
        int e = p + 1, depth = 1;
        while (e < body.length() && depth > 0) {
            if (body.charAt(e) == '[') depth++;
            else if (body.charAt(e) == ']') depth--;
            e++;
        }
        return parseVec(body.substring(p + 1, e - 1));
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new HashMap<>();
        if (query == null) return map;
        for (String param : query.split("&")) {
            String[] kv = param.split("=", 2);
            if (kv.length == 2) {
                map.put(URLDecoder.decode(kv[0], StandardCharsets.UTF_8),
                        URLDecoder.decode(kv[1], StandardCharsets.UTF_8));
            }
        }
        return map;
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    static void addCors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    static void sendJson(HttpExchange ex, int status, String json) throws IOException {
        addCors(ex);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    // =====================================================================
    //  DEMO DATA
    // =====================================================================

    static void loadDemo(VectorDB db) {
        DistFn dist = getDistFn("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }

    // =====================================================================
    //  MAIN HTTP SERVER
    // =====================================================================

    public static void main(String[] args) throws Exception {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        Pattern deletePattern = Pattern.compile("^/delete/(\\d+)$");
        Pattern docDeletePattern = Pattern.compile("^/doc/delete/(\\d+)$");

        server.createContext("/", ex -> {
            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();

            if ("OPTIONS".equalsIgnoreCase(method)) {
                addCors(ex);
                ex.sendResponseHeaders(204, -1);
                ex.close();
                return;
            }

            try {
                // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────
                if ("GET".equals(method) && "/search".equals(path)) {
                    Map<String, String> params = parseQuery(ex.getRequestURI().getRawQuery());
                    float[] q = parseVec(params.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = 5;
                    try { if (params.containsKey("k")) k = Integer.parseInt(params.get("k")); } catch (Exception ignored) {}
                    String metric = params.getOrDefault("metric", "cosine");
                    String algo = params.getOrDefault("algo", "hnsw");
                    sendJson(ex, 200, db.searchJson(q, k, metric, algo));
                }
                else if ("POST".equals(method) && "/insert".equals(path)) {
                    String body = readBody(ex);
                    String meta = extractStr(body, "metadata");
                    String cat = extractStr(body, "category");
                    float[] emb = extractArr(body, "embedding");
                    if (meta.isEmpty() || emb.length != DIMS) {
                        sendJson(ex, 200, "{\"error\":\"invalid body\"}");
                        return;
                    }
                    int id = db.insert(meta, cat, emb, getDistFn("cosine"));
                    sendJson(ex, 200, "{\"id\":" + id + "}");
                }
                else if ("DELETE".equals(method) && deletePattern.matcher(path).matches()) {
                    Matcher m = deletePattern.matcher(path);
                    m.find();
                    int id = Integer.parseInt(m.group(1));
                    boolean ok = db.remove(id);
                    sendJson(ex, 200, "{\"ok\":" + ok + "}");
                }
                else if ("GET".equals(method) && "/items".equals(path)) {
                    List<VectorItem> items = db.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < items.size(); i++) {
                        if (i > 0) ss.append(',');
                        VectorItem v = items.get(i);
                        ss.append("{\"id\":").append(v.id)
                          .append(",\"metadata\":").append(jS(v.metadata))
                          .append(",\"category\":").append(jS(v.category))
                          .append(",\"embedding\":").append(jVec(v.emb)).append('}');
                    }
                    ss.append(']');
                    sendJson(ex, 200, ss.toString());
                }
                else if ("GET".equals(method) && "/benchmark".equals(path)) {
                    Map<String, String> params = parseQuery(ex.getRequestURI().getRawQuery());
                    float[] q = parseVec(params.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = 5;
                    try { if (params.containsKey("k")) k = Integer.parseInt(params.get("k")); } catch (Exception ignored) {}
                    String metric = params.getOrDefault("metric", "cosine");
                    sendJson(ex, 200, db.benchmarkJson(q, k, metric));
                }
                else if ("GET".equals(method) && "/hnsw-info".equals(path)) {
                    sendJson(ex, 200, db.hnswInfoJson());
                }

                // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────
                else if ("POST".equals(method) && "/doc/insert".equals(path)) {
                    String body = readBody(ex);
                    String title = extractStr(body, "title");
                    String text = extractStr(body, "text");
                    if (title.isEmpty() || text.isEmpty()) {
                        sendJson(ex, 200, "{\"error\":\"need title and text\"}");
                        return;
                    }
                    List<String> chunks = chunkText(text, 250, 30);
                    List<Integer> ids = new ArrayList<>();
                    for (int i = 0; i < chunks.size(); i++) {
                        float[] emb = ollama.embed(chunks.get(i));
                        if (emb.length == 0) {
                            sendJson(ex, 200, "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                            return;
                        }
                        String chunkTitle = (chunks.size() > 1)
                                ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                                : title;
                        ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
                    }
                    String idsStr = ids.toString().replace(" ", "");
                    sendJson(ex, 200, "{\"ids\":" + idsStr + ",\"chunks\":" + chunks.size() + ",\"dims\":" + docDB.dims + "}");
                }
                else if ("DELETE".equals(method) && docDeletePattern.matcher(path).matches()) {
                    Matcher m = docDeletePattern.matcher(path);
                    m.find();
                    int id = Integer.parseInt(m.group(1));
                    boolean ok = docDB.remove(id);
                    sendJson(ex, 200, "{\"ok\":" + ok + "}");
                }
                else if ("GET".equals(method) && "/doc/list".equals(path)) {
                    List<DocItem> docs = docDB.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < docs.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocItem d = docs.get(i);
                        String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                        int words = d.text.trim().isEmpty() ? 0 : d.text.trim().split("\\s+").length;
                        ss.append("{\"id\":").append(d.id)
                          .append(",\"title\":").append(jS(d.title))
                          .append(",\"preview\":").append(jS(preview))
                          .append(",\"words\":").append(words).append('}');
                    }
                    ss.append(']');
                    sendJson(ex, 200, ss.toString());
                }
                else if ("POST".equals(method) && "/doc/search".equals(path)) {
                    String body = readBody(ex);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(ex, 200, "{\"error\":\"need question\"}");
                        return;
                    }
                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}");
                        return;
                    }
                    List<DocHit> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ss = new StringBuilder("{\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocHit h = hits.get(i);
                        ss.append("{\"id\":").append(h.doc.id)
                          .append(",\"title\":").append(jS(h.doc.title))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.dist)).append('}');
                    }
                    ss.append("]}");
                    sendJson(ex, 200, ss.toString());
                }
                else if ("POST".equals(method) && "/doc/ask".equals(path)) {
                    String body = readBody(ex);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(ex, 200, "{\"error\":\"need question\"}");
                        return;
                    }
                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}");
                        return;
                    }
                    List<DocHit> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ctx = new StringBuilder();
                    for (int i = 0; i < hits.size(); i++) {
                        ctx.append("[").append(i + 1).append("] ").append(hits.get(i).doc.title).append(":\n")
                           .append(hits.get(i).doc.text).append("\n\n");
                    }
                    String prompt =
                        "You are a helpful assistant. Answer the user's question directly. " +
                        "Use the provided context if it contains relevant information. " +
                        "If it doesn't, just use your own general knowledge. " +
                        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. " +
                        "Just answer the question naturally.\n\n" +
                        "Context:\n" + ctx +
                        "Question: " + question + "\n\n" +
                        "Answer:";

                    String answer = ollama.generate(prompt);
                    StringBuilder ss = new StringBuilder("{\"answer\":");
                    ss.append(jS(answer))
                      .append(",\"model\":").append(jS(ollama.genModel))
                      .append(",\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocHit h = hits.get(i);
                        ss.append("{\"id\":").append(h.doc.id)
                          .append(",\"title\":").append(jS(h.doc.title))
                          .append(",\"text\":").append(jS(h.doc.text))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.dist)).append('}');
                    }
                    ss.append("],\"docCount\":").append(docDB.size()).append('}');
                    sendJson(ex, 200, ss.toString());
                }
                else if ("GET".equals(method) && "/status".equals(path)) {
                    boolean up = ollama.isAvailable();
                    String json = "{\"ollamaAvailable\":" + up +
                                  ",\"embedModel\":" + jS(ollama.embedModel) +
                                  ",\"genModel\":" + jS(ollama.genModel) +
                                  ",\"docCount\":" + docDB.size() +
                                  ",\"docDims\":" + docDB.dims +
                                  ",\"demoDims\":" + DIMS +
                                  ",\"demoCount\":" + db.size() + "}";
                    sendJson(ex, 200, json);
                }
                else if ("GET".equals(method) && "/stats".equals(path)) {
                    String json = "{\"count\":" + db.size() +
                                  ",\"dims\":" + DIMS +
                                  ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]" +
                                  ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
                    sendJson(ex, 200, json);
                }
                else if ("GET".equals(method) && "/".equals(path)) {
                    Path htmlPath = Path.of("index.html");
                    if (!Files.exists(htmlPath)) {
                        ex.sendResponseHeaders(404, -1);
                        ex.close();
                        return;
                    }
                    byte[] html = Files.readAllBytes(htmlPath);
                    ex.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
                    ex.sendResponseHeaders(200, html.length);
                    try (OutputStream os = ex.getResponseBody()) {
                        os.write(html);
                    }
                }
                else {
                    ex.sendResponseHeaders(404, -1);
                    ex.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
                sendJson(ex, 500, "{\"error\":" + jS(e.getMessage()) + "}");
            }
        });

        server.start();
    }
}