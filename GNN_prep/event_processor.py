import duckdb
from typing import Dict
from config import Config
import numpy as np
import pandas as pd

class EventProcessor:
    """
    Class for the pre-process of the multi-event file.
    This version saves both 'base_relevance' and 'adjusted_score' for evaluation.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, config: Config):
        """
        Args:
            con: duckdb connection
            config: global Config object
        """
        self.con = con

        self.embeddings_path = config.paths.audio_embeddings_file
        self.multi_event_path = config.paths.raw_multi_event_file

        self.low_threshold = config.preprocessing.low_interaction_threshold
        self.high_threshold = config.preprocessing.high_interaction_threshold

        self.split_ratios = config.preprocessing.split_ratios
        self.split_paths = config.paths.split_paths

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

        self.top_k = config.gnn.top_popular_k
        self.weight = config.preprocessing.weights
        self.novelty = config.preprocessing.novelty


    def _compute_active_users(self):
        """
        finds all the users that have more interactions than the given threshold.
        creates a temporary table 'active_users' in the DuckDB memory.
        """
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
        """
        filters the multi-event file out of songs without provided embeddings and non-active users.
        creates a temporary table 'filtered_event' in the DuckDB memory.
        """
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
        """
        encodes the given user IDs into GNN-ready IDs (continuous integers).
        creates a temporary table 'events_with_idx' in the DuckDB memory.
        """
        query = """
            CREATE TEMPORARY TABLE events_with_idx AS
            WITH 
            user_index AS (
                SELECT uid, ROW_NUMBER() OVER (ORDER BY uid) - 1 AS user_id
                FROM (SELECT DISTINCT uid FROM filtered_events)
            )
            SELECT e.*, u.user_id
            FROM filtered_events e
            JOIN user_index u USING (uid)
            """
        self.con.execute(query)
        print("Created user indices")


    def filter_events(self, low_threshold: int = None, high_threshold: int = None, output_path: str = None ):
        """
        runs the multi-event filtering pipeline:
            1. find active users with more interactions than the given threshold
            2. filter out songs and users
            3. encode user IDs

        Add a threshold value to override the config-defined threshold
        and/or an output path to save the filtered file as a parquet.

        Args:
            low_threshold: lower threshold for interactions, default: none
            high_threshold: higher threshold for interactions, default: none
            output_path: path to save the filtered file, default: none
        """
        if low_threshold is not None:
            self.low_threshold = low_threshold

        if high_threshold is not None:
            self.high_threshold = high_threshold

        self._compute_active_users()
        self._filter_multi_event_file()
        self._encode_user_ids()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM events_with_idx) TO '{output_path}' (FORMAT PARQUET)")
            print(f'Filtered multi event file saved to {output_path}')


    def _split_data(self):
        """
        splits the filtered multi-event file into train, validation and test sets
        """
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
                       WHEN o.rn <= {self.split_ratios['train']} * o.total_events THEN 'train'
                       WHEN o.rn <= ({self.split_ratios['train']} + {self.split_ratios['val']}) * o.total_events THEN 'val'
                       ELSE 'test'
                   END AS split
            FROM ordered o
            ORDER BY o.user_id, o.rn
        """

        self.con.execute(query)
        print(f"\nData was split into:\n"
              f"{self.split_ratios['train'] * 100}% train set\n"
              f"{self.split_ratios['val'] * 100}% validation set\n"
              f"{self.split_ratios['test'] * 100}% test set\n")


    def _save_cold_start_songs(self):
        """
        save the audio embeddings and song IDs of songs not in the train set for evaluation purposes.
        """
        self.con.execute("""
            CREATE TEMPORARY TABLE test_items AS
            SELECT DISTINCT item_id 
            FROM split_data 
            WHERE split = 'test'
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE cold_start_songs AS
            SELECT d.item_id, emb.normalized_embed
            FROM split_data d
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON d.item_id = emb.item_id
            LEFT JOIN test_items t
                ON d.item_id = t.item_id
            WHERE d.split IN ('train', 'val') 
              AND t.item_id IS NULL
        """)

        self.con.execute(f"""COPY (SELECT * FROM cold_start_songs) TO '{self.cold_start_songs_path}' (FORMAT PARQUET)""")
        print(f'Cold start songs file saved to {self.cold_start_songs_path}')


    def _remove_neg_train_edges(self):
        """
        Removes negative interactions from the train set
        """
        query = f"""
            CREATE TEMPORARY TABLE no_neg_train_events AS
            SELECT *
            FROM split_data 
            WHERE event_type IN ('listen', 'like', 'undislike')
            AND split = 'train'
        """

        self.con.execute(query)
        print("Finished removing negative edges")


    def _save_neg_interactions(self):
        """
        Saves negative interactions for a given split into two files:
        1. In-Graph: Lightweight file with user_id, item_id (for items in the train set).
        2. Cold-Start: Heavy file with user_id, item_id, normalized_embed (for items NOT in the train set).
        """
        split = 'train'

        print(f"Splitting negative '{split}' interactions...")

        # 1. Get all unique item_ids that are in the 'train' split (positive or negative)
        # These are the items that will be "in-graph"
        self.con.execute("""
            CREATE TEMPORARY TABLE train_set_items AS
            SELECT DISTINCT item_id
            FROM split_data
            WHERE split = 'train'
        """)

        # 2. Get all negative events for the target split
        self.con.execute(f"""
            CREATE TEMPORARY TABLE neg_events AS
            SELECT DISTINCT uid, user_id, item_id, event_type
            FROM split_data
            WHERE event_type NOT IN ('listen', 'like', 'undislike')
              AND split = '{split}'
        """)

        # 3. Save In-Graph negatives (lightweight)
        # These are negative events where the item_id *IS* in the train_set_items
        in_graph_query = f"""
            COPY (
                SELECT neg.user_id, neg.item_id
                FROM neg_events AS neg
                INNER JOIN train_set_items AS tsi
                  ON neg.item_id = tsi.item_id
            ) TO '{self.negative_train_in_graph_file}' (FORMAT PARQUET)
        """
        self.con.execute(in_graph_query)
        print(f"Saved in-graph negative interactions to {self.negative_train_in_graph_file}")

        # 4. Save Cold-Start negatives (heavy)
        # These are negative events where the item_id *IS NOT* in the train_set_items
        cold_start_query = f"""
            COPY (
                SELECT neg.user_id, neg.item_id, emb.normalized_embed
                FROM neg_events AS neg
                LEFT JOIN train_set_items AS tsi
                  ON neg.item_id = tsi.item_id
                INNER JOIN read_parquet('{self.embeddings_path}') AS emb
                  ON neg.item_id = emb.item_id
                WHERE tsi.item_id IS NULL
            ) TO '{self.negative_train_cold_start_file}' (FORMAT PARQUET)
        """
        self.con.execute(cold_start_query)
        print(f"Saved cold-start negative interactions to {self.negative_train_cold_start_file}")


    def _save_splits(self):
        self.con.execute(f"COPY (SELECT * FROM no_neg_train_events) TO '{self.split_paths['train']}' (FORMAT PARQUET)")
        print(f"Train data saved to {self.split_paths['train']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='val') TO '{self.split_paths['val']}' (FORMAT PARQUET)")
        print(f"Validation data saved to {self.split_paths['val']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='test') TO '{self.split_paths['test']}' (FORMAT PARQUET)")
        print(f"Test data saved to {self.split_paths['test']}")


    def _process_split(self, split: str, case_base: str, novelty_params: dict, out_path: str):
        """
        Runs the full SQL pipeline for a single data split (val or test).
        1. Create {split}_raw (event-level scores)
        2. Create {split}_scores (user-item aggregated scores)
        3. Create {split}_final (novelty-adjusted scores)
        4. Export to Parquet
        """
        # Use a shorter alias for f-strings
        n = novelty_params

        # --- 1. Event-level scores ---
        raw_query = f"""
            CREATE TEMPORARY TABLE {split}_raw AS
            SELECT
                e.user_id,
                e.item_id,
                e.timestamp,
                ( {case_base} )                                AS base_relevance,
                1                                              AS n_events,
                COALESCE(t.play_cnt,0)                         AS train_play_cnt,
                COALESCE(t.seen_in_train,0)                    AS seen_in_train
            FROM split_data e
            LEFT JOIN (
                SELECT user_id, item_id,
                       COUNT(*)               AS play_cnt,
                       1                      AS seen_in_train
                FROM split_data
                WHERE split = 'train' AND event_type = 'listen'
                GROUP BY user_id, item_id
            ) t
                ON e.user_id = t.user_id AND e.item_id = t.item_id
            WHERE e.split = '{split}'
        """
        self.con.execute(raw_query)

        # --- 2. Aggregated scores per user-item ---
        agg_query = f"""
            CREATE TEMPORARY TABLE {split}_scores AS
            SELECT
                user_id,
                item_id,
                SUM(base_relevance)    AS base_relevance,
                SUM(n_events)          AS total_events,
                MAX(seen_in_train)     AS seen_in_train,
                MAX(train_play_cnt)    AS train_play_cnt,
                MAX(timestamp)         AS latest_ts
            FROM {split}_raw
            GROUP BY user_id, item_id
        """
        self.con.execute(agg_query)

        # --- 3. Final scores with novelty adjustment ---
        final_query = f"""
            CREATE TEMPORARY TABLE {split}_final AS
            SELECT
                user_id,
                item_id,
                base_relevance,
                total_events,
                seen_in_train,
                train_play_cnt,

                -- Calculate novelty factor
                (CASE WHEN seen_in_train = 0 THEN {n['unseen_boost']} ELSE 0 END)
                - {n['train_penalty']} * LEAST(train_play_cnt, {n['max_familiarity']}) / {n['max_familiarity']}
                -- + CASE WHEN seen_in_train = 1 THEN
                --         EXP(-{n['recency_beta']} * (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) - latest_ts))
                --      ELSE 0 END
                AS novelty_factor,

                -- Calculate final adjusted score
                (base_relevance * (1 + novelty_factor))^2 AS adjusted_score

            FROM {split}_scores
        """
        self.con.execute(final_query)

        # --- 4. Export results to Parquet ---
        export_query = f"""
            COPY (
                SELECT user_id, item_id, base_relevance, adjusted_score, 
                       total_events, seen_in_train, train_play_cnt 
                FROM {split}_final
            ) TO '{out_path}' (FORMAT PARQUET)
        """
        self.con.execute(export_query)

        print(f"  > {split.capitalize()} scores saved → {out_path}")


    def _build_relevance_case_statement(self, weights: Dict[str, float]) -> str:
        """
        Builds the SQL CASE statement string for calculating base relevance
        based on event type and weights.
        """
        case_base = "CASE e.event_type\n"
        for etype, weight in weights.items():
            if etype == "listen":
                case_base += f"    WHEN '{etype}' THEN {weight} * ((COALESCE(e.played_ratio_pct,0)/100.0))\n"
            else:
                case_base += f"    WHEN '{etype}' THEN {weight}\n"
        case_base += "    ELSE 0.0 END"
        return case_base


    def _compute_relevance_scores(self):
        """
        Main coordinator function.
        Calculates and saves base and novelty-adjusted relevance scores
        for validation and test splits.
        """
        # 1. Get attributes from self
        w = self.weight
        n = self.novelty

        # 2. Build reusable SQL snippet
        case_base = self._build_relevance_case_statement(w)

        # 3. Process each split sequentially
        print("Processing validation split...")
        self._process_split(
            split='val',
            case_base=case_base,
            novelty_params=n,
            out_path=self.val_scores # Assuming paths are still in Config
        )

        print("Processing test split...")
        self._process_split(
            split='test',
            case_base=case_base,
            novelty_params=n,
            out_path=self.test_scores  # Assuming paths are still in Config
        )
        print("Relevance score processing complete.")


    def split_data(self, split_ratios: dict = None):
        """
        splits the filtered multi-event file into train, validation and test sets and
        saves the embeddings of the cold-start songs.

        Args:
            split_ratios: dictionary of split ratios, default: none
        """
        if split_ratios is not None:
            self.split_ratios = split_ratios

        self._split_data()
        self._compute_relevance_scores()
        self._save_cold_start_songs()
        self._remove_neg_train_edges()
        self._save_neg_interactions()
        self._save_splits()


    def _save_filtered_user_ids(self):
        """
        Save sorted list of all filtered/encoded user IDs (0 to num_users-1) as npy.
        """
        output_path = self.filtered_user_ids

        query = f"""
            SELECT DISTINCT user_id
            FROM read_parquet('{self.split_paths['train']}')
            ORDER BY user_id
        """
        df = self.con.execute(query).fetch_df()
        user_ids = df['user_id'].to_numpy(dtype=np.int64)
        np.save(output_path, user_ids)
        print(f"Saved {len(user_ids)} filtered user IDs to {output_path}")


    def _save_filtered_song_ids(self):
        """
        Save sorted list of all unique filtered song IDs (global item_id) as npy.
        """
        output_path = self.filtered_song_ids

        query = f"""
            WITH unique_items AS (
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['train']}')
                UNION
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['val']}')
                UNION
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['test']}')
            )
            SELECT item_id FROM unique_items ORDER BY item_id
        """
        df = self.con.execute(query).fetch_df()
        song_ids = df['item_id'].to_numpy(dtype=np.int64)
        np.save(output_path, song_ids)
        print(f"Saved {len(song_ids)} filtered song IDs to {output_path}")


    def _save_filtered_audio_embeddings(self):
        """
        Save audio embeddings for all filtered songs as parquet (item_id, normalized_embed).
        """
        output_path = self.filtered_audio_embed_file

        query = f"""
            CREATE OR REPLACE TEMPORARY TABLE filtered_songs_emb AS
            WITH unique_items AS (
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['train']}')
                UNION
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['val']}')
                UNION
                SELECT DISTINCT item_id FROM read_parquet('{self.split_paths['test']}')
            )
            SELECT ui.item_id, emb.normalized_embed
            FROM unique_items ui
            JOIN read_parquet('{self.embeddings_path}') emb ON ui.item_id = emb.item_id
            ORDER BY ui.item_id
        """
        self.con.execute(query)
        self.con.execute(f"COPY filtered_songs_emb TO '{output_path}' (FORMAT PARQUET)")
        print(
            f"Saved filtered audio embeddings ({self.con.execute('SELECT COUNT(*) FROM filtered_songs_emb').fetchone()[0]} songs) to {output_path}")


    def _save_most_popular_songs(self):
        """
        Save top-K most popular song IDs (based on count of positive interactions in train) as npy.
        """
        output_path = self.popular_song_ids

        query = f"""
            SELECT item_id
            FROM read_parquet('{self.split_paths['train']}') e
            WHERE e.event_type IN ('listen', 'like', 'undislike')
            GROUP BY item_id
            ORDER BY COUNT(*) DESC
            LIMIT {self.top_k}
        """
        df = self.con.execute(query).fetch_df()
        popular_ids = df['item_id'].to_numpy(dtype=np.int64)
        np.save(output_path, popular_ids)
        print(f"Saved top-{self.top_k} popular song IDs to {output_path}")


    def _save_positive_interactions(self):
        """
        Save unique positive user-item pairs for a split as parquet (user_id, item_id).
        """
        output_path = self.positive_interactions_file
        split_name = 'train'

        split_path = self.split_paths[split_name]
        temp_table = f"{split_name}_positive_interactions"
        query = f"""
            CREATE OR REPLACE TEMPORARY TABLE {temp_table} AS
            SELECT DISTINCT user_id, item_id
            FROM read_parquet('{split_path}')
            WHERE event_type IN ('listen', 'like', 'undislike')
            ORDER BY user_id, item_id
        """
        self.con.execute(query)
        self.con.execute(f"COPY {temp_table} TO '{output_path}' (FORMAT PARQUET)")
        count = self.con.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        print(f"Saved {count} {split_name} positive interactions to {output_path}")


    def _compute_user_avg_embeddings(self):
        """
        Compute per-user average audio embedding over distinct positive items in a split.
        """
        output_path = self.filtered_user_embed_file
        split_name = 'train'

        split_path = self.split_paths[split_name]
        temp_table = f"{split_name}_user_pos_emb"
        query = f"""
            CREATE OR REPLACE TEMPORARY TABLE {temp_table} AS
            SELECT e.user_id, e.item_id, ANY_VALUE(emb.normalized_embed) AS normalized_embed
            FROM read_parquet('{split_path}') e
            JOIN read_parquet('{self.embeddings_path}') emb ON e.item_id = emb.item_id
            WHERE e.event_type IN ('listen', 'like', 'undislike')
            GROUP BY e.user_id, e.item_id
        """
        self.con.execute(query)

        df = self.con.execute(f"SELECT user_id, normalized_embed FROM {temp_table}").fetch_df()

        avg_embs = []
        for user_id, group in df.groupby("user_id"):
            embs_list = list(group["normalized_embed"])
            embs_array = np.vstack(embs_list)  # Stack to [num_items, dim]
            avg_emb = np.mean(embs_array, axis=0)
            avg_embs.append({"user_id": int(user_id), "avg_embed": avg_emb.tolist()})

        avg_df = pd.DataFrame(avg_embs).sort_values("user_id")
        avg_df.to_parquet(output_path, index=False)
        print(f"Saved {len(avg_embs)} {split_name} user average embeddings to {output_path}")


    def prepare_baselines(self):
        """
        Prepare all necessary files for baseline models.
        """
        self._save_filtered_user_ids()
        self._save_filtered_song_ids()
        self._save_filtered_audio_embeddings()
        self._save_most_popular_songs()
        self._save_positive_interactions()
        self._compute_user_avg_embeddings()

