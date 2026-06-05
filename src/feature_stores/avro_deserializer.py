import io
import json
import logging
import struct

from config import SCHEMA_REGISTRY_URL

logger = logging.getLogger("PyFlinkCDCFeast")


class AvroDecoder:
    """Decode Confluent wire-format Avro bytes to JSON string.

    Used as a helper inside CDCParser (FlatMapFunction).
    Confluent wire format: [0x00][4-byte schema_id][avro_payload]

    Schema is fetched lazily from Schema Registry per schema_id and cached.
    __getstate__/__setstate__ let PyFlink pickle this safely across workers.
    """

    def __init__(self):
        self._schema_cache: dict = {}
        self._registry = None

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._schema_cache = {}
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            from confluent_kafka.schema_registry import SchemaRegistryClient
            self._registry = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})
        return self._registry

    def _get_schema(self, schema_id: int):
        if schema_id not in self._schema_cache:
            import fastavro
            schema_str = self._get_registry().get_schema(schema_id).schema_str
            self._schema_cache[schema_id] = fastavro.parse_schema(json.loads(schema_str))
        return self._schema_cache[schema_id]

    def decode(self, raw: str) -> str:
        """Convert ISO-8859-1 encoded string (from Kafka) → JSON string.

        Kafka messages are read as ISO-8859-1 strings to preserve binary bytes.
        If the first byte is 0x00 (Confluent magic byte) → Avro decode.
        Otherwise return as-is (plain JSON fallback).
        """
        try:
            raw_bytes = raw.encode("ISO-8859-1")
        except Exception:
            return raw

        if len(raw_bytes) < 5 or raw_bytes[0] != 0:
            return raw  # plain JSON — pass through

        try:
            import fastavro
            _, schema_id = struct.unpack(">bI", raw_bytes[:5])
            schema = self._get_schema(schema_id)
            record = fastavro.schemaless_reader(io.BytesIO(raw_bytes[5:]), schema)
            return json.dumps(record, default=str)
        except Exception as exc:
            logger.warning("Avro decode failed: %s", exc)
            return ""
