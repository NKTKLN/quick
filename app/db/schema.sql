CREATE TABLE IF NOT EXISTS function_calls (
    timestamp TIMESTAMP,
    function_name TEXT,
    args BLOB,
    kwargs BLOB,
    result BLOB
);
