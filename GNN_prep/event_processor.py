import duckdb
from typing import Dict
from config import Config
import numpy as np
import pandas as pd


class EventProcessor:
    """
    Class for the pre-process of the multi-event file.
    Refactored for modularity and readability.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, config: Config):
        self.con = con

        self.embeddings_path = config.paths.audio_embeddings_file
        self.multi_event_path = config.paths.raw_multi_event_file
        self.val_scores = config.paths.val_scores_file
        self.test_scores = config.paths.test_scores_file
        self.cold_start_songs_path = config.paths.cold_start_songs_file
        self.filtered_audio_embed_file = config.paths.filtered_audio_embed_file
        self.filtered_user_embed_file = config.paths.filtered_user_embed_file
        self.filtered_song_ids = config.paths.filtered_song_ids
        self.filtered_user_ids = config.paths.filtered_user_ids
        self.popular_song_ids = config.paths.popular_song_ids
        self.positive_interactions_file = config.paths.positive_interactions_file
        self.negative_train_in_graph_file = config.paths.negative_train_in_graph_file
        self.negative_train_cold_start_file = config.paths.negative_train_cold_start_file
        self.split_paths = config.paths.split_paths

        self.low_threshold = config.preprocessing.low_interaction_threshold
        self.high_threshold = config.preprocessing.high_interaction_threshold
        self.split_ratios = config.preprocessing.split_ratios
        self.top_k = config.gnn.top_popular_k
        self.weight = config.preprocessing.weights
        self.novelty = config.preprocessing.novelty

    # ---------------------------------------------------------
    # 1. Filtering & User Encoding
    # ---------------------------------------------------------

    def _compute_active_users(self):
        """Identifies users within the interaction threshold."""
        query = f"""
            CREATE TEMPORARY TABLE active_users AS
            SELECT uid
            FROM read_parquet('{self.multi_event_path}')
            GROUP BY uid
            HAVING COUNT(*) >= {self.low_threshold}
            AND COUNT(*) <= {self.high_threshold}
        """
        self.con.execute(query)
        print("Found all active users")


    def _filter_multi_event_file(self):
        """Filters events to keep only active users and items with embeddings."""
        query = f"""
            CREATE TEMPORARY TABLE filtered_events AS
            SELECT e.*
            FROM read_parquet('{self.multi_event_path}') e
            INNER JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            INNER JOIN active_users au
                ON e.uid = au.uid
            WHERE e.uid IS NOT NULL AND e.item_id IS NOT NULL
        """
        self.con.execute(query)
        print("Finished filtering the multi-event interactions")


    def _encode_user_ids(self):
        """Maps original UIDs to continuous 0-indexed integers."""
        query = """
            CREATE TEMPORARY TABLE events_with_idx AS
            WITH user_index AS (
                SELECT uid, ROW_NUMBER() OVER (ORDER BY uid) - 1 AS user_id
                FROM (SELECT DISTINCT uid FROM filtered_events)
            )
            SELECT e.*, u.user_id
            FROM filtered_events e
                JOIN user_index u USING (uid) 
        """
        self.con.execute(query)
        print("Created user indices")


    def filter_events(self, low_threshold: int = None, high_threshold: int = None, output_path: str = None):
        """Main entry point for filtering."""
        if low_threshold is not None: self.low_threshold = low_threshold
        if high_threshold is not None: self.high_threshold = high_threshold

        self._compute_active_users()
        self._filter_multi_event_file()
        self._encode_user_ids()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM events_with_idx) TO '{output_path}' (FORMAT PARQUET)")
            print(f'Filtered multi event file saved to {output_path}')

    # ---------------------------------------------------------
    # 2. Data Splitting
    # ---------------------------------------------------------

    def _split_data(self):
        """Splits data into train/val/test based on timestamp order per user."""
        train_ratio = self.split_ratios['train']
        val_limit = train_ratio + self.split_ratios['val']

        query = f"""
            CREATE TEMPORARY TABLE split_data AS
            WITH ordered AS (
                SELECT e.*,
                       ROW_NUMBER() OVER (PARTITION BY e.user_id ORDER BY e.timestamp) AS rn,
                       COUNT(*) OVER (PARTITION BY e.user_id) AS total_events
                FROM events_with_idx e
            )
            SELECT o.*,
                   CASE 
                       WHEN o.rn <= {train_ratio} * o.total_events THEN 'train'
                       WHEN o.rn <= {val_limit} * o.total_events THEN 'val'
                       ELSE 'test'
                   END AS split
            FROM ordered o
            ORDER BY o.user_id, o.rn
        """
        self.con.execute(query)
        print(
            f"Data split: {train_ratio * 100}% Train, {self.split_ratios['val'] * 100}% Val, {self.split_ratios['test'] * 100}% Test")


    def _save_splits(self):
        """Exports the split datasets to parquet."""
        for split_name, path in self.split_paths.items():
            # For train, we use the filtered 'no_neg_train_events' table created later
            if split_name == 'train':
                source_table = "no_neg_train_events"
            else:
                source_table = f"split_data WHERE split='{split_name}'"

            self.con.execute(f"COPY (SELECT * FROM {source_table}) TO '{path}' (FORMAT PARQUET)")
            print(f"{split_name.capitalize()} data saved to {path}")


    def _remove_neg_train_edges(self):
        """Filters out negative interactions from the training set (used for graph construction)."""
        query = """
                CREATE \
                TEMPORARY TABLE no_neg_train_events AS
                SELECT *
                FROM split_data
                WHERE event_type IN ('listen', 'like', 'undislike')
                  AND split = 'train' \
                """
        self.con.execute(query)
        print("Finished removing negative edges from train set")

    # ---------------------------------------------------------
    # 3. Negative Interaction Processing
    # ---------------------------------------------------------

    def _create_negative_event_tables(self, split: str = 'train'):
        """Creates temporary tables for negative events and train items."""
        # 1. Identify all items present in the training set
        self.con.execute("""
            CREATE TEMPORARY TABLE train_set_items AS
            SELECT DISTINCT item_id FROM split_data WHERE split = 'train'
        """)

        # 2. Isolate negative events
        self.con.execute(f"""
            CREATE TEMPORARY TABLE neg_events AS
            SELECT DISTINCT uid, user_id, item_id, event_type
            FROM split_data
            WHERE event_type NOT IN ('listen', 'like', 'undislike')
              AND split = '{split}'
        """)


    def _save_in_graph_negatives(self):
        """Saves negatives where the item IS in the training graph (lightweight)."""
        query = f"""
            COPY (
                SELECT neg.user_id, neg.item_id
                FROM neg_events AS neg
                INNER JOIN train_set_items AS tsi ON neg.item_id = tsi.item_id
            ) TO '{self.negative_train_in_graph_file}' (FORMAT PARQUET)
        """
        self.con.execute(query)
        print(f"Saved in-graph negative interactions.")


    def _save_cold_start_negatives(self):
        """Saves negatives where the item IS NOT in the training graph (heavy, needs embeddings)."""
        query = f"""
            COPY (
                SELECT neg.user_id, neg.item_id, emb.normalized_embed
                FROM neg_events AS neg
                LEFT JOIN train_set_items AS tsi ON neg.item_id = tsi.item_id
                INNER JOIN read_parquet('{self.embeddings_path}') AS emb ON neg.item_id = emb.item_id
                WHERE tsi.item_id IS NULL
            ) TO '{self.negative_train_cold_start_file}' (FORMAT PARQUET)
        """
        self.con.execute(query)
        print(f"Saved cold-start negative interactions.")


    def _save_neg_interactions(self):
        """Coordinator for processing negative interactions."""
        print(f"Splitting negative 'train' interactions...")
        self._create_negative_event_tables(split='train')
        self._save_in_graph_negatives()
        self._save_cold_start_negatives()

        # Cleanup temp tables if needed, though DuckDB handles temp tables per connection
        self.con.execute("DROP TABLE IF EXISTS train_set_items")
        self.con.execute("DROP TABLE IF EXISTS neg_events")

    # ---------------------------------------------------------
    # 4. Cold Start & Relevance Processing
    # ---------------------------------------------------------

    def _save_cold_start_songs(self):
        """Identifies and saves songs present in Test but missing from Train/Val."""
        # Identify test items
        self.con.execute(
            "CREATE TEMPORARY TABLE test_items AS SELECT DISTINCT item_id FROM split_data WHERE split = 'test'")

        # Find cold start items and fetch embeddings
        self.con.execute(f"""
            CREATE TEMPORARY TABLE cold_start_songs AS
            SELECT d.item_id, emb.normalized_embed
            FROM split_data d
            LEFT JOIN read_parquet('{self.embeddings_path}') emb ON d.item_id = emb.item_id
            LEFT JOIN test_items t ON d.item_id = t.item_id
            WHERE d.split IN ('train', 'val') AND t.item_id IS NULL
        """)

        self.con.execute(f"COPY (SELECT * FROM cold_start_songs) TO '{self.cold_start_songs_path}' (FORMAT PARQUET)")
        print(f'Cold start songs file saved.')


    def _build_relevance_case_statement(self) -> str:
        """Constructs the SQL CASE statement for base relevance."""
        case_base = "CASE e.event_type\n"
        for etype, weight in self.weight.items():
            if etype == "listen":
                case_base += f"    WHEN '{etype}' THEN {weight} * ((COALESCE(e.played_ratio_pct,0)/100.0))\n"
            else:
                case_base += f"    WHEN '{etype}' THEN {weight}\n"
        case_base += "    ELSE 0.0 END"
        return case_base


    def _create_raw_split_table(self, split: str, case_base: str):
        """Step 1: Create event-level scores."""
        query = f"""
            CREATE TEMPORARY TABLE {split}_raw AS
            SELECT
                e.user_id, e.item_id, e.timestamp,
                ( {case_base} ) AS base_relevance,
                1 AS n_events,
                COALESCE(t.play_cnt,0) AS train_play_cnt,
                COALESCE(t.seen_in_train,0) AS seen_in_train
            FROM split_data e
            LEFT JOIN (
                SELECT user_id, item_id, COUNT(*) AS play_cnt, 1 AS seen_in_train
                FROM split_data
                WHERE split = 'train' AND event_type = 'listen'
                GROUP BY user_id, item_id
            ) t ON e.user_id = t.user_id AND e.item_id = t.item_id
            WHERE e.split = '{split}'
        """
        self.con.execute(query)


    def _create_aggregated_score_table(self, split: str):
        """Step 2: Aggregate scores per user-item."""
        query = f"""
            CREATE TEMPORARY TABLE {split}_scores AS
            SELECT
                user_id, item_id,
                SUM(base_relevance) AS base_relevance,
                SUM(n_events) AS total_events,
                MAX(seen_in_train) AS seen_in_train,
                MAX(train_play_cnt) AS train_play_cnt,
                MAX(timestamp) AS latest_ts
            FROM {split}_raw
            GROUP BY user_id, item_id
        """
        self.con.execute(query)


    def _create_final_score_table(self, split: str):
        """Step 3: Apply novelty adjustment."""
        n = self.novelty
        query = f"""
            CREATE TEMPORARY TABLE {split}_final AS
            SELECT
                user_id, item_id, base_relevance, total_events, seen_in_train, train_play_cnt,
                (CASE WHEN seen_in_train = 0 THEN {n['unseen_boost']} ELSE 0 END)
                - {n['train_penalty']} * LEAST(train_play_cnt, {n['max_familiarity']}) / {n['max_familiarity']}
                AS novelty_factor,
                (base_relevance * (1 + novelty_factor))^2 AS adjusted_score
            FROM {split}_scores
        """
        self.con.execute(query)


    def _process_split(self, split: str, out_path: str):
        """Orchestrates the scoring pipeline for a specific split."""
        case_base = self._build_relevance_case_statement()

        self._create_raw_split_table(split, case_base)
        self._create_aggregated_score_table(split)
        self._create_final_score_table(split)

        # Step 4: Export
        export_query = f"""
            COPY (
                SELECT user_id, item_id, base_relevance, adjusted_score, 
                       total_events, seen_in_train, train_play_cnt 
                FROM {split}_final
            ) TO '{out_path}' (FORMAT PARQUET)
        """
        self.con.execute(export_query)
        print(f"  > {split.capitalize()} scores processed and saved.")

        # Cleanup
        self.con.execute(f"DROP TABLE {split}_raw")
        self.con.execute(f"DROP TABLE {split}_scores")
        self.con.execute(f"DROP TABLE {split}_final")

    def _compute_relevance_scores(self):
        """Computes scores for both Val and Test splits."""
        print("Processing validation scores...")
        self._process_split('val', self.val_scores)

        print("Processing test scores...")
        self._process_split('test', self.test_scores)

    # ---------------------------------------------------------
    # 5. Main Pipeline Methods
    # ---------------------------------------------------------

    def split_data(self, split_ratios: dict = None):
        """Main entry point for splitting and processing pipeline."""
        if split_ratios is not None: self.split_ratios = split_ratios

        self._split_data()
        self._compute_relevance_scores()
        self._save_cold_start_songs()
        self._remove_neg_train_edges()
        self._save_neg_interactions()
        self._save_splits()

    # ---------------------------------------------------------
    # 6. Baseline Preparation (Helpers)
    # ---------------------------------------------------------

    def _save_filtered_user_ids(self):
        query = f"SELECT DISTINCT user_id FROM read_parquet('{self.split_paths['train']}') ORDER BY user_id"
        df = self.con.execute(query).fetch_df()
        np.save(self.filtered_user_ids, df['user_id'].to_numpy(dtype=np.int64))


    def _save_filtered_song_ids(self):
        query = f"""
            WITH unique_items AS (
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['train']}')
                UNION SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['val']}')
                UNION SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['test']}')
            ) SELECT item_id FROM unique_items ORDER BY item_id
        """
        df = self.con.execute(query).fetch_df()
        np.save(self.filtered_song_ids, df['item_id'].to_numpy(dtype=np.int64))


    def _save_filtered_audio_embeddings(self):
        query = f"""
            CREATE OR REPLACE TEMPORARY TABLE filtered_songs_emb AS
            WITH unique_items AS (
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['train']}')
                UNION SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['val']}')
                UNION SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['test']}')
            )
            SELECT ui.item_id, emb.normalized_embed
            FROM unique_items ui
            JOIN read_parquet('{self.embeddings_path}') emb ON ui.item_id = emb.item_id
            ORDER BY ui.item_id
        """
        self.con.execute(query)
        self.con.execute(f"COPY filtered_songs_emb TO '{self.filtered_audio_embed_file}' (FORMAT PARQUET)")


    def _save_most_popular_songs(self):
        query = f"""
            SELECT item_id FROM read_parquet('{self.split_paths['train']}')
            WHERE event_type IN ('listen', 'like', 'undislike')
            GROUP BY item_id ORDER BY COUNT(*) DESC LIMIT {self.top_k}
        """
        df = self.con.execute(query).fetch_df()
        np.save(self.popular_song_ids, df['item_id'].to_numpy(dtype=np.int64))


    def _save_positive_interactions(self):
        query = f"""
            CREATE OR REPLACE TEMPORARY TABLE train_pos_interactions AS
            SELECT DISTINCT user_id, item_id
            FROM read_parquet('{self.split_paths['train']}')
            WHERE event_type IN ('listen', 'like', 'undislike')
            ORDER BY user_id, item_id
        """
        self.con.execute(query)
        self.con.execute(f"COPY train_pos_interactions TO '{self.positive_interactions_file}' (FORMAT PARQUET)")


    def _compute_user_avg_embeddings(self):
        # Re-using the temp table from _save_positive_interactions logic if available, or querying again
        query = f"""
            SELECT e.user_id, e.item_id, ANY_VALUE(emb.normalized_embed) AS normalized_embed
            FROM read_parquet('{self.split_paths['train']}') e
            JOIN read_parquet('{self.embeddings_path}') emb ON e.item_id = emb.item_id
            WHERE e.event_type IN ('listen', 'like', 'undislike')
            GROUP BY e.user_id, e.item_id
        """
        df = self.con.execute(query).fetch_df()

        # Numpy aggregation (faster/easier than DuckDB for array averaging sometimes)
        avg_embs = []
        for user_id, group in df.groupby("user_id"):
            embs_array = np.vstack(list(group["normalized_embed"]))
            avg_emb = np.mean(embs_array, axis=0)
            avg_embs.append({"user_id": int(user_id), "avg_embed": avg_emb.tolist()})

        pd.DataFrame(avg_embs).sort_values("user_id").to_parquet(self.filtered_user_embed_file, index=False)


    def prepare_baselines(self):
        print("\nPreparing baseline files...")
        self._save_filtered_user_ids()
        self._save_filtered_song_ids()
        self._save_filtered_audio_embeddings()
        self._save_most_popular_songs()
        self._save_positive_interactions()
        self._compute_user_avg_embeddings()
        print("Baseline files prepared.")

