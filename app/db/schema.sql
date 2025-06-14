CREATE TABLE IF NOT EXISTS function_calls (
    timestamp TIMESTAMP,
    function_name TEXT,
    self_data BLOB,
    args BLOB,
    kwargs BLOB,
    result BLOB
);
