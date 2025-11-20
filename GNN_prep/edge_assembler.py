import duckdb
from config import Config

class EdgeAssembler:
    """
    Class to assemble the data for the graph construction.
    Refactored for modularity.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, config: Config):
        self.con = con
        self.train_path = config.paths.train_set_file
        self.weights = config.preprocessing.weights
        self.embeddings_path = config.paths.audio_embeddings_file
        self.album_mapping_path = config.paths.album_mapping_file
        self.artist_mapping_path = config.paths.artist_mapping_file
        self.event_type_mapping = config.preprocessing.edge_type_mapping


    def _build_weight_case_stmt(self) -> str:
        """Constructs SQL CASE statement for edge weights."""
        stmt = "CASE e.event_type\n"
        for etype, weight in self.weights.items():
            stmt += f"    WHEN '{etype}' THEN {weight}\n"
        stmt += "END AS edge_weight"
        return stmt


    def _build_event_type_case_stmt(self) -> str:
        """Constructs SQL CASE statement for numeric edge types."""
        stmt = "CASE e.event_type\n"
        for etype, cat in self.event_type_mapping.items():
            stmt += f"    WHEN '{etype}' THEN {cat}\n"
        stmt += "    ELSE 0\nEND AS edge_type"
        return stmt


    def _aggregate_edges(self):
        """
        Aggregates interactions by user-song-event.
        Calculates counts, weights, and average play ratios.
        """
        case_weight = self._build_weight_case_stmt()
        case_event_type = self._build_event_type_case_stmt()

        query = f"""
            CREATE TEMPORARY TABLE agg_edges AS
            SELECT 
                e.uid, e.user_id, e.item_id, e.event_type, 
                COUNT(*) AS edge_count,
                {case_weight},
                {case_event_type},
                AVG(
                    CASE 
                        WHEN e.event_type = 'listen' THEN LEAST(e.played_ratio_pct, 100.0) / 100.0
                        WHEN e.event_type = 'like' THEN 1
                        WHEN e.event_type = 'dislike' THEN 0
                        ELSE 0.5
                    END
                ) AS edge_avg_played_ratio,
                ANY_VALUE(emb.normalized_embed) AS item_normalized_embed
            FROM read_parquet('{self.train_path}') e
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            GROUP BY e.uid, e.user_id, e.item_id, e.event_type
        """
        self.con.execute(query)
        print("Finished aggregating the edges")


    def _prepare_train_item_map(self):
        """Creates a mapping from global item_id to continuous train indices (0..N)."""
        self.con.execute(f"""
            CREATE TEMPORARY TABLE train_items AS
            SELECT DISTINCT item_id
            FROM read_parquet('{self.train_path}')
            WHERE split = 'train'
        """)

        self.con.execute("""
             CREATE TEMPORARY TABLE train_item_map AS
             SELECT item_id, ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_train_idx
             FROM train_items
        """)
        print("Prepared train item re-index mapping")


    def _prepare_artist_album_metadata(self):
        """Encodes artist and album IDs and prepares metadata tables."""
        # Create simple 0-indexed tables for artists and albums
        self.con.execute(f"""
            CREATE TEMPORARY TABLE artist_index AS
            SELECT artist_id, ROW_NUMBER() OVER (ORDER BY artist_id) AS artist_idx
            FROM (SELECT DISTINCT artist_id FROM read_parquet('{self.artist_mapping_path}'))
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE album_index AS
            SELECT album_id, ROW_NUMBER() OVER (ORDER BY album_id) AS album_idx
            FROM (SELECT DISTINCT album_id FROM read_parquet('{self.album_mapping_path}'))
        """)

        # Map songs to these new indices
        self.con.execute(f"""
            CREATE TEMPORARY TABLE song_artist_meta AS
            SELECT s.item_id, s.artist_id, a.artist_idx
            FROM read_parquet('{self.artist_mapping_path}') s
            LEFT JOIN artist_index a USING (artist_id)
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE song_album_meta AS
            SELECT s.item_id, s.album_id, b.album_idx
            FROM read_parquet('{self.album_mapping_path}') s
            LEFT JOIN album_index b USING (album_id)
        """)
        print("Prepared artist and album metadata")


    def _join_edges_with_metadata(self):
        """Joins aggregated edges with the prepared metadata and train indices."""
        self.con.execute("""
            CREATE TEMPORARY TABLE merged_raw AS
            SELECT ae.item_id,
                COALESCE(tm.item_train_idx, -1) AS item_train_idx,
                ae.item_normalized_embed,
                COALESCE(am.artist_id, 0)       AS artist_id,
                COALESCE(am.artist_idx, 0)      AS artist_idx,
                COALESCE(al.album_id, 0)        AS album_id,
                COALESCE(al.album_idx, 0)       AS album_idx
            FROM agg_edges ae
                LEFT JOIN song_artist_meta am ON ae.item_id = am.item_id
                LEFT JOIN song_album_meta al ON ae.item_id = al.item_id
                LEFT JOIN train_item_map tm ON ae.item_id = tm.item_id
        """)


    def _deduplicate_metadata(self):
        """
        Deduplicates metadata.
        Since edges are user-song-event specific, the metadata (artist/album)
        should be constant per song. We group by item_id to ensure unique metadata per item.
        """
        self.con.execute("""
            CREATE TEMPORARY TABLE agg_edges_artist_album AS
            SELECT 
                item_id, item_train_idx,
                ANY_VALUE(item_normalized_embed) AS item_normalized_embed,
                MAX(artist_idx)                  AS artist_idx,
                MAX(album_idx)                   AS album_idx,
                MAX(artist_id)                   AS artist_id,
                MAX(album_id)                    AS album_id
            FROM merged_raw
            GROUP BY item_id, item_train_idx
        """)
        print("Merged edges and metadata, deduplicated, and re-indexed items for train")


    def _add_song_metadata(self):
        """Coordinator for adding metadata to edges."""
        self._prepare_train_item_map()
        self._prepare_artist_album_metadata()
        self._join_edges_with_metadata()
        self._deduplicate_metadata()
        # Cleanup intermediate table
        self.con.execute("DROP TABLE merged_raw")


    def assemble_edges(self, output_path: str = None):
        """
        Runs the full edge assembler pipeline:
            1. aggregate the edges
            2. add artist and album info

        Args:
            output_path: path to output file, default: None
        """
        self._aggregate_edges()
        self._add_song_metadata()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM agg_edges_artist_album) TO '{output_path}' (FORMAT PARQUET)")
            print(f"Edge data saved to {output_path}")