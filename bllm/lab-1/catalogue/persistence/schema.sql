CREATE TABLE products (
    id INTEGER PRIMARY KEY CHECK (id > 0),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    category TEXT NOT NULL CHECK (category IN ('laptop', 'smartphone', 'tablet')),
    price INTEGER NOT NULL CHECK (price >= 0)
) STRICT;
