CREATE TABLE IF NOT EXISTS function_calls (
    timestamp TIMESTAMP,
    function_name TEXT,
    key_data BLOB,
    result BLOB
);