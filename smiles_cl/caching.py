import hashlib
import pickle
from abc import ABC, abstractmethod
from functools import _make_key, cached_property
from pathlib import Path
from typing import Optional


class StorageError(Exception):
    pass


class BaseCacheStorage(ABC):
    @abstractmethod
    def store(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, key):
        raise NotImplementedError


class DictCacheStorage(BaseCacheStorage):
    def __init__(self):
        self._cache = {}

    def store(self, key, value):
        self._cache[key] = value

    def retrieve(self, key):
        return self._cache[key]


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(x.bit_length() + 1, "big", signed=True)


class LMDBCacheStorage(BaseCacheStorage):
    """Cache storage using an LMDB as backend"""

    def __init__(
        self,
        db_path: str,
        map_size: int = 64 * 2**30,
        readonly: bool = False,
        serializer=pickle,
    ):
        try:
            import lmdb

            self.lmdb = lmdb
        except ImportError:
            raise ImportError("lmdb package must be installed to use LMDBCacheStorage")

        self._env = None
        self._txn = None

        self.db_path = str(db_path)
        self.map_size = map_size
        self.readonly = readonly
        self.serializer = serializer

    @property
    def env(self):
        if self._env is None:
            self._env = self.lmdb.open(
                self.db_path, map_size=self.map_size, readonly=self.readonly
            )
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._txn = self.env.begin(write=not self.readonly)
        return self._txn

    def get_db_size(self):
        stats = self.env.stat()

        # See https://stackoverflow.com/a/40527056
        db_size = stats["psize"] * (
            stats["leaf_pages"] + stats["branch_pages"] + stats["overflow_pages"] + 1
        )

        return db_size

    def _get_free_space(self):
        return self.map_size - self.get_db_size()

    def _hash_key(self, key):
        return hashlib.sha256(int_to_bytes(hash(key))).digest()

    def store(self, key, value):
        assert (
            not self.readonly
        ), "Storage is configured as readonly; cannot store values"

        key_bytes = self._hash_key(key)
        value_bytes = self.serializer.dumps(value)

        if len(value_bytes) > self._get_free_space():
            raise StorageError("Value is too large to be stored in the database")

        self.txn.put(key_bytes, value_bytes)

    def get(self, raw_key):
        value = self.txn.get(raw_key)

        if value is None:
            raise KeyError(f"Key not found: {raw_key}")

        return self.serializer.loads(value)

    def retrieve(self, key):
        return self.get(self._hash_key(key))


class SqliteCacheStorage(BaseCacheStorage):
    """Cache storage using Sqlite as backend"""

    def __init__(self, db_path: str, serializer=pickle):
        try:
            import sqlitedict

            self.sqlitedict = sqlitedict
        except ImportError:
            raise ImportError(
                "sqlitedict package must be installed to use SqliteCacheStorage"
            )

        self.db_path = str(db_path)
        self.serializer = serializer

    @cached_property
    def _db(self):
        return self.sqlitedict.SqliteDict(
            self.db_path,
            autocommit=True,
            encode=self.serializer.dumps,
            decode=self.serializer.loads,
        )

    def store(self, key, value):
        self._db[hash(key)] = value

    def retrieve(self, key):
        return self._db[hash(key)]


class cache:
    def __init__(self, storage: Optional[BaseCacheStorage] = None, typed: bool = True):
        if storage is None:
            storage = DictCacheStorage()

        self.storage = storage
        self.typed = typed

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            key = _make_key(args, kwargs, typed=self.typed)

            try:
                return self.storage.retrieve(key)
            except KeyError:
                value = fn(*args, **kwargs)
                self.storage.store(key, value)

                return value

        return wrapped


def get_lmdb_cache():
    storage = LMDBCacheStorage(Path(__file__).parent / "cache.lmdb")
    return cache(storage=storage)
