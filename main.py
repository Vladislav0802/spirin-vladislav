"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""



def create_submission(submission):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import ndcg_score
    
    import re
    import random
    
    import math
    from collections import Counter
    
    from sentence_transformers import SentenceTransformer
    import torch
    from tqdm import tqdm
    
    import optuna
    from catboost import CatBoostRanker, Pool


    RANDOM_SEED = 993
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    train = pd.read_csv('train.csv')
    train.head()

    def descriptive_statistics(df):
        print(df.shape)
        print(df.info())
        print(df.describe())
        print(df.describe(include='object'))
        print(df[df.duplicated()].sum())
        print(df.isna().sum())
        if 'relevance' in df.columns:
            print(df['relevance'].value_counts())

    descriptive_statistics(train)

    test = pd.read_csv('test.csv')
    test.head()

    descriptive_statistics(test)

    fill_map = {
    "product_description": "",
    "product_bullet_point": "",
    "product_brand": "unknown_brand",
    "product_color": "unknown_color",
    "product_title": ""
    }
    def fill_none(df):
        df = df.drop(columns=['product_locale'], errors='ignore')
        df['product_description'] = df['product_description'].replace('none', "")
        for col, val in fill_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    train = fill_none(train)

    fill_none(test)
    descriptive_statistics(train)

    test = fill_none(test)
    descriptive_statistics(test)

    _WORD_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
    STOP = set(ENGLISH_STOP_WORDS)

    def tokenize(text, lowercase=True, remove_stop=True, min_len=2):
        if not isinstance(text, str):
            return []
        if lowercase:
            text = text.lower()
        toks = _WORD_RE.findall(text)
        if remove_stop:
            toks = [t for t in toks if t not in STOP and len(t) >= min_len]
        else:
            toks = [t for t in toks if len(t) >= min_len]
        return toks

    def prepare_text_data(df):
        text_cols = [
            "query",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color"
        ]
        for c in text_cols:
            df[c + "_tok"] = df[c].apply(lambda x: tokenize(x))
    
        df["product_all_tok"] = (
            df["product_title_tok"]
            + df["product_description_tok"]
            + df["product_bullet_point_tok"]
            + df["product_brand_tok"]
            + df["product_color_tok"]
        )
    
        df["query_clean"] = df["query_tok"].apply(lambda toks: " ".join(toks))
        df["text_all"] = df["product_all_tok"].apply(lambda toks: " ".join(toks))
    
        return df[[
            "id",
            "query_id",
            "query_clean",
            "text_all"
        ] + (["relevance"] if "relevance" in df.columns else [])]

    def add_rank_features_improved(df):
        df["query_len"] = df["query_clean"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        df["title_len"] = df["text_all"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    
        def overlap(q, t):
            if not isinstance(q, str) or not isinstance(t, str):
                return 0
            q_set = set(q.split())
            t_set = set(t.split())
            return len(q_set & t_set)
        
        df["overlap_q_title"] = df.apply(lambda row: overlap(row["query_clean"], row["text_all"]), axis=1)
        
        def normalized_overlap(q, t):
            if not isinstance(q, str) or not isinstance(t, str):
                return 0
            q_set = set(q.split())
            t_set = set(t.split())
            if len(q_set) == 0:
                return 0
            return len(q_set & t_set) / len(q_set)
        
        df["overlap_norm"] = df.apply(lambda row: normalized_overlap(row["query_clean"], row["text_all"]), axis=1)
        
        def jaccard(q, t):
            if not isinstance(q, str) or not isinstance(t, str):
                return 0
            q_set = set(q.split())
            t_set = set(t.split())
            denom = len(q_set | t_set)
            return len(q_set & t_set) / denom if denom > 0 else 0
        
        df["jaccard_q_text"] = df.apply(lambda row: jaccard(row["query_clean"], row["text_all"]), axis=1)
        
        def coverage(q, t):
            if not isinstance(q, str) or not isinstance(t, str):
                return 0
            q_set = set(q.split())
            t_set = set(t.split())
            if len(t_set) == 0:
                return 0
            return len(q_set & t_set) / len(t_set)
        
        df["coverage"] = df.apply(lambda row: coverage(row["query_clean"], row["text_all"]), axis=1)
        df["length_diff"] = df.apply(lambda row: abs(row["query_len"] - row["title_len"]), axis=1)
        
        numeric_features = ["query_len", "title_len", "overlap_q_title", "overlap_norm", 
                           "jaccard_q_text", "coverage", "length_diff"]
        
        return df, numeric_features

    def compute_bm25_scores(df, doc_col="text_all", query_col="query_clean",
                        k1=1.5, b=0.75):
        docs = [doc.split() if isinstance(doc, str) else [] for doc in df[doc_col].values]
        N = len(docs)
    
        df_counts = Counter()
        lens = []
        for tokens in docs:
            lens.append(len(tokens))
            df_counts.update(set(tokens))
        avgdl = np.mean(lens) if len(lens) > 0 else 0.0
    
        idf = {}
        for term, df_t in df_counts.items():
            idf_val = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9)
            idf[term] = idf_val
    
        scores = np.zeros(N, dtype=float)
        for i, (doc_tokens, query_text) in enumerate(zip(docs, df[query_col].values)):
            if not isinstance(query_text, str) or len(doc_tokens) == 0:
                scores[i] = 0.0
                continue
            q_terms = query_text.split()
            if len(q_terms) == 0:
                scores[i] = 0.0
                continue
            tf = Counter(doc_tokens)
            dl = len(doc_tokens)
            denom_const = k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else k1
            s = 0.0
            for term in q_terms:
                if term not in tf or term not in idf:
                    continue
                f = tf[term]
                numer = f * (k1 + 1)
                denom = f + denom_const
                s += idf[term] * numer / denom
            scores[i] = s
    
        return scores

    def add_sbert_features(df, model_name="sentence-t5-base", batch_size=64):

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = SentenceTransformer(model_name, device=device)
    
        df["query_clean"] = df["query_clean"].fillna("").astype(str)
        df["text_all"] = df["text_all"].fillna("").astype(str)
    
        def embed_texts(texts):
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="SBERT embedding"):
                batch = texts[i:i+batch_size]
                embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device=device)
                embeddings.append(embs)
            return np.vstack(embeddings)
    
        query_embs = embed_texts(df["query_clean"].tolist())
        text_embs = embed_texts(df["text_all"].tolist())
    
        cos_sims = []
        for i in tqdm(range(0, len(df), batch_size), desc="Cosine similarity"):
            q_batch = query_embs[i:i+batch_size]
            t_batch = text_embs[i:i+batch_size]
            sim_batch = np.sum(q_batch * t_batch, axis=1) / (np.linalg.norm(q_batch, axis=1) * np.linalg.norm(t_batch, axis=1))
            cos_sims.extend(sim_batch)
    
        df["sbert_cos_sim"] = cos_sims
        numeric_features = ["sbert_cos_sim"]
        
        return df, numeric_features

    def split_rank_data(df, group_col="query_id", test_size=0.2, random_state=RANDOM_SEED):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        
        groups = df[group_col]
    
        train_idx, valid_idx = next(splitter.split(df, groups=groups))
    
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
    
        return train_df, valid_df
    
    
    def make_rank_pool_with_numeric(df, numeric_features=None):
        text_features = ["query_clean", "text_all"]
        data = df[text_features].copy()
        if numeric_features:
            for n in numeric_features:
                data[n] = df[n].values
        pool = Pool(
            data=data,
            label=df["relevance"],
            group_id=df["query_id"],
            text_features=text_features
        )
        return pool

    df_prepared = prepare_text_data(train)
    df_prepared["bm25_score"] = compute_bm25_scores(df_prepared, doc_col="text_all", query_col="query_clean")
    df_prepared, numeric_feats = add_rank_features_improved(df_prepared)
    numeric_feats.append("bm25_score")
    df_prepared, sbert_feats = add_sbert_features(df_prepared)
    numeric_feats.extend(sbert_feats)
    
    train_df, valid_df = split_rank_data(df_prepared)
    
    train_pool = make_rank_pool_with_numeric(train_df, numeric_features=numeric_feats)
    valid_pool = make_rank_pool_with_numeric(valid_df, numeric_features=numeric_feats)

    # def objective(trial):
    #     params = {
    #         "loss_function": "YetiRank",
    #         "eval_metric": "NDCG:top=10",
    #         "text_features": ["query_clean", "text_all"],
    #         "feature_calcers": ["BoW"],
    #         "tokenizers": [{
    #             'tokenizer_id': 'Space',
    #             'separator_type': 'ByDelimiter',
    #             'delimiter': ' '
    #         }],
    #         "iterations": trial.suggest_int("iterations", 300, 800),
    #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    #         "depth": trial.suggest_int("depth", 4, 10),
    #         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
    #         "random_seed": RANDOM_SEED,
    #         "thread_count": -1,
    #         "use_best_model": True,
    #         "verbose": 0
    #     }
        
    #     model = CatBoostRanker(**params)
    #     model.fit(train_pool, eval_set=valid_pool, verbose=0)
        
    #     metrics = model.eval_metrics(valid_pool, metrics=["NDCG:top=10"], ntree_end=model.tree_count_)
    #     best_score = metrics["NDCG:top=10;type=Base"][-1]
        
    #     return best_score
    
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=10) 
    
    # print("Лучшие параметры:", study.best_params)
    # print("Лучший NDCG@10:", study.best_value)

    best_params = {'iterations': 611, 'learning_rate': 0.025431479532075488, 'depth': 9, 'l2_leaf_reg': 4.2051628592584915}
    best_params.update({
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "text_features": ["query_clean", "text_all"],
        "feature_calcers": ["BoW"],
        "tokenizers": [{
            'tokenizer_id': 'Space',
            'separator_type': 'ByDelimiter',
            'delimiter': ' '
        }],
        "random_seed": RANDOM_SEED,
        "thread_count": -1,
        "use_best_model": True,
        "verbose": 50
    })

    final_ranker = CatBoostRanker(**best_params)
    final_ranker.fit(train_pool, eval_set=valid_pool)

    valid_df["prediction"] = final_ranker.predict(valid_pool)

    def compute_ndcg(df, qid_col="query_id", rel_col="relevance", pred_col="prediction", k=10):
        ndcgs = []
        for qid, group in df.groupby(qid_col):
            y_true = group[rel_col].values.reshape(1, -1)
            y_pred = group[pred_col].values.reshape(1, -1)
            ndcgs.append(ndcg_score(y_true, y_pred, k=k))
        return np.mean(ndcgs)
    
    val_ndcg = compute_ndcg(valid_df)
    print("Validation nDCG@10 =", val_ndcg)

    def make_rank_pool_test(df, numeric_features=None):
        text_features = ["query_clean", "text_all"]
        data = df[text_features].copy()
    
        if numeric_features:
            for n in numeric_features:
                data[n] = df[n].values
        
        pool = Pool(
            data=data,
            group_id=df["query_id"],
            text_features=text_features
        )
        return pool

    test_prepared = prepare_text_data(test)

    test_prepared["bm25_score"] = compute_bm25_scores(
        test_prepared, 
        doc_col="text_all", 
        query_col="query_clean"
    )
    test_prepared, numeric_feats_test = add_rank_features_improved(test_prepared)
    numeric_feats_test.append("bm25_score")
    test_prepared, sbert_feats_test = add_sbert_features(test_prepared)
    numeric_feats_test.extend(sbert_feats_test)
    
    test_pool = make_rank_pool_test(test_prepared, numeric_features=numeric_feats_test)
    
    test_prepared["prediction"] = final_ranker.predict(test_pool)

    submission = test_prepared[["id", "prediction"]]

    
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
