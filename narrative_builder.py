import argparse
import json
import os
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import nltk

nltk.data.path.append("C:\\Users\\nivas\\nltk_data")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def load_json_file(path):
    print(f"Loading dataset from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # CASE 1: dataset is {"items": [...articles...]}
    if isinstance(data, dict) and "items" in data:
        articles = data["items"]
        return articles

    # CASE 2: dataset is already a list
    if isinstance(data, list):
        return data

    # Otherwise invalid
    raise ValueError("Unsupported JSON format: expected a list or an 'items' field.")


def filter_by_rating(articles, min_rating=8.0):
    """
    Keep articles whose 'source_rating' exists and is >= min_rating.
    Strict interpretation: missing rating will be excluded.
    """
    filtered = []
    for a in articles:
        rating = a.get("source_rating")
        try:
            if rating is not None and float(rating) > min_rating:
                filtered.append(a)
        except:
            # If rating cannot be parsed, skip it (strict)
            continue

    print(f"Filtered {len(filtered)} articles with source_rating > {min_rating} (from {len(articles)})")
    return filtered


def normalize_text(item):
    # Prefer 'story', then 'text', then 'content'
    title = (item.get("title") or item.get("headline") or "").strip()
    text = (item.get("story") or item.get("text") or item.get("content") or "").strip()
    if not text and "body" in item:
        # Sometimes nested
        if isinstance(item["body"], str):
            text = item["body"]
    combined = (title + "\n\n" + text).strip()
    # fallback to url or summary if nothing else
    if not combined:
        combined = item.get("url", "") or item.get("summary", "") or ""
    return combined


def parse_date(item):
    # try multiple fields: date, published_at, pubDate
    for k in ("date", "published_at", "publishedAt", "pubDate", "created_at"):
        if k in item and item[k]:
            try:
                dt = dateparser.parse(item[k])
                if dt is None:
                    continue
                # Convert offset-aware -> naive
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except:
                pass
    return None


def embed_texts(model, texts, batch_size=64):
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        e = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs.append(e)
    return np.vstack(embs)


def retrieve_relevant_articles(topic, corpus_embeddings, corpus_articles, model, top_k=150):
    if corpus_embeddings.shape[0] == 0:
        return [], [], []

    q_emb = model.encode([topic], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, corpus_embeddings)[0]
    ranked_idx = np.argsort(-sims)[:min(top_k, len(sims))]
    selected = [corpus_articles[i] for i in ranked_idx]
    scores = [float(sims[i]) for i in ranked_idx]
    return selected, ranked_idx, scores


def build_summary(selected_texts, max_sentences=7):
    """
    Very simple extractive summary:
    - Take first sentence of each article
    - Keep the longest 5–max_sentences sentences
    """
    sentences = []
    for text in selected_texts:
        if not text:
            continue
        try:
            sents = nltk.sent_tokenize(text)
            if sents:
                sentences.append(sents[0])
        except:
            # fallback naive split by period
            pieces = [p.strip() for p in text.split(".") if p.strip()]
            if pieces:
                sentences.append(pieces[0] + ".")
    # If not enough sentences, take more from articles
    if len(sentences) < max_sentences:
        for text in selected_texts:
            try:
                sents = nltk.sent_tokenize(text)
                sentences.extend(sents[:2])
            except:
                continue
    # Deduplicate and sort by length (longer -> often informative)
    seen = set()
    unique_sents = []
    for s in sentences:
        s_clean = " ".join(s.split())
        if s_clean not in seen:
            seen.add(s_clean)
            unique_sents.append(s_clean)
    unique_sents = sorted(unique_sents, key=lambda s: -len(s))
    selected = unique_sents[:max_sentences]
    summary = " ".join(selected)
    if not summary:
        summary = "No summary could be produced from the selected articles."
    return summary


def build_timeline(selected_articles, scores):
    timeline = []
    for art, score in zip(selected_articles, scores):
        date = parse_date(art)
        timeline.append({
            "date": date.isoformat() if date else None,
            "headline": art.get("title") or art.get("headline") or "",
            "url": art.get("url") or art.get("link") or None,
            "why_it_matters": make_why_it_matters(art),
            "score": float(score)
        })
    timeline_sorted = sorted(timeline, key=lambda x: (x["date"] is None, x["date"]))
    return timeline_sorted


def make_why_it_matters(article):
    title = article.get("title", "") or ""
    text = article.get("story") or article.get("text") or article.get("content") or ""
    first_sent = ""
    try:
        sents = nltk.sent_tokenize(text)
        if sents:
            first_sent = sents[0]
    except:
        first_sent = text[:200]
    why = (title + ". " + first_sent).strip()
    if len(why) > 300:
        why = why[:297] + "..."
    # Clean whitespace
    return " ".join(why.split())


def cluster_and_label(selected_texts, n_clusters=5):
    if len(selected_texts) == 0:
        return [], []
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(selected_texts)
    n_clusters = min(n_clusters, max(1, len(selected_texts)))
    if n_clusters == 1:
        labels = np.zeros(len(selected_texts), dtype=int)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    # label clusters by top tf-idf terms
    cluster_info = []
    try:
        if n_clusters > 1:
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = tfidf.get_feature_names_out()
            for c in clusters:
                top_terms = [terms[ind] for ind in order_centroids[c, :8] if ind < len(terms)]
                label = ", ".join(top_terms[:4])
                cluster_info.append({
                    "cluster_id": int(c),
                    "label": label,
                    "article_indices": clusters[c]
                })
        else:
            # Single cluster label: top tf-idf terms across all docs
            terms = tfidf.get_feature_names_out()
            # pick top 6 terms by average score
            avg = np.asarray(X.mean(axis=0)).ravel()
            top_idx = avg.argsort()[::-1][:6]
            label = ", ".join([terms[i] for i in top_idx if i < len(terms)])
            cluster_info.append({
                "cluster_id": 0,
                "label": label,
                "article_indices": clusters[0]
            })
    except Exception:
        # Fallback simple labels
        for c in clusters:
            cluster_info.append({
                "cluster_id": int(c),
                "label": f"cluster_{c}",
                "article_indices": clusters[c]
            })
    return labels, cluster_info


def build_graph(selected_texts, selected_articles, model, threshold=0.6):
    """
    Build directed graph:
      nodes = articles (index in selected list)
      edges = if semantic similarity >= threshold
      relation = builds_on if target date > source date else context_for or related_to
    """
    if len(selected_texts) == 0:
        return {"nodes": [], "edges": []}

    embs = model.encode(selected_texts, convert_to_numpy=True)
    sim = cosine_similarity(embs)
    G = nx.DiGraph()
    for i, art in enumerate(selected_articles):
        date = parse_date(art)
        G.add_node(i, title=art.get("title", ""), url=art.get("url", ""), date=date)

    n = len(selected_articles)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            score = float(sim[i, j])
            if score >= threshold:
                di = G.nodes[i].get("date")
                dj = G.nodes[j].get("date")
                label = "related_to"
                if di and dj:
                    try:
                        if dj > di:
                            label = "builds_on"
                        elif dj < di:
                            label = "context_for"
                    except Exception:
                        label = "related_to"
                G.add_edge(i, j, weight=score, relation=label)

    nodes = [
        {
            "id": int(nid),
            "title": G.nodes[nid].get("title"),
            "url": G.nodes[nid].get("url"),
            "date": G.nodes[nid].get("date").isoformat() if G.nodes[nid].get("date") else None
        } for nid in G.nodes()
    ]
    edges = [
        {
            "source": int(u),
            "target": int(v),
            "weight": G[u][v]["weight"],
            "relation": G[u][v]["relation"]
        } for u, v in G.edges()
    ]
    return {"nodes": nodes, "edges": edges}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, type=str, help="Topic to build narrative for")
    parser.add_argument("--datafile", required=True, type=str, help="Path to news JSON file")
    parser.add_argument("--top_k", type=int, default=150, help="Number of top relevant articles to use")
    parser.add_argument("--min_rating", type=float, default=8.0, help="Filter source_rating >= min_rating")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()

    # 1) Load data
    articles = load_json_file(args.datafile)

    # 2) Filter by rating
    articles_filtered = filter_by_rating(articles, min_rating=args.min_rating)
    if len(articles_filtered) == 0:
        print(json.dumps({"error": "no articles after rating filter"}))
        return

    # 3) Build corpus texts and metadata (use normalize_text that prefers 'story')
    corpus_texts = [normalize_text(a) for a in articles_filtered]

    # 4) Load model once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5) Embed corpus
    corpus_embeddings = embed_texts(model, corpus_texts)

    # 6) Retrieve relevant articles
    selected_articles, ranked_idx, scores = retrieve_relevant_articles(
        args.topic, corpus_embeddings, articles_filtered, model, top_k=args.top_k
    )
    if len(selected_articles) == 0:
        print(json.dumps({"error": "no relevant articles found for the topic"}))
        return

    selected_texts = [normalize_text(a) for a in selected_articles]

    # 7) Build narrative summary
    narrative_summary = build_summary(selected_texts, max_sentences=7)

    # 8) Build timeline
    timeline = build_timeline(selected_articles, scores)

    # 9) Clusters
    labels, cluster_info = cluster_and_label(selected_texts, n_clusters=args.clusters)

    clusters_output = []
    for c in cluster_info:
        idxs = c["article_indices"]
        clusters_output.append({
            "cluster_id": c["cluster_id"],
            "label": c["label"],
            "articles": [
                {
                    "index_in_selected": int(i),
                    "title": selected_articles[i].get("title", ""),
                    "url": selected_articles[i].get("url", ""),
                    "date": parse_date(selected_articles[i]).isoformat() if parse_date(selected_articles[i]) else None
                } for i in idxs
            ]
        })

    # 10) Narrative graph (re-uses the same model)
    graph = build_graph(selected_texts, selected_articles, model, threshold=0.6)

    # 11) Final JSON
    out = {
        "narrative_summary": narrative_summary,
        "timeline": timeline,
        "clusters": clusters_output,
        "graph": graph
    }

    # Print pretty JSON to stdout (single output only)
    print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
