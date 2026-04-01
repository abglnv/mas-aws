import psycopg2
import psycopg2.extras
import logging
from app.config import settings

log = logging.getLogger(__name__)


def get_conn():
    return psycopg2.connect(settings.postgres_url)


def execute_query(sql: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]


def get_schema() -> str:
    return """
Tables in the database:

users(id SERIAL, name VARCHAR, email VARCHAR, created_at TIMESTAMP)
transactions(id SERIAL, user_id INT, amount DECIMAL, status VARCHAR, product_id INT, created_at TIMESTAMP)
product_catalog(id SERIAL, name VARCHAR, price DECIMAL, category VARCHAR, stock INT, description TEXT)

Foreign keys: transactions.user_id -> users.id, transactions.product_id -> product_catalog.id
"""


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS product_catalog (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    category VARCHAR(50),
                    stock INT DEFAULT 0,
                    description TEXT
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    id SERIAL PRIMARY KEY,
                    user_id INT REFERENCES users(id),
                    product_id INT REFERENCES product_catalog(id),
                    amount DECIMAL(10,2) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] > 0:
                log.info("Database already seeded, skipping.")
                return

            cur.execute("""
                INSERT INTO users (name, email) VALUES
                ('Aibek Dzhaksybekov', 'aibek.dzhaksybekov@mail.ru'),
                ('Dinara Seitkali', 'dinara_s@yandex.kz'),
                ('Yerlan Abenov', 'yerlan.abenov@gmail.com'),
                ('Madina Nurlanovna', 'madina.nur@inbox.kz'),
                ('Dauren Bekzatov', 'dauren.b@mail.ru');

                INSERT INTO product_catalog (name, price, category, stock, description) VALUES
                ('Laptop Pro X1', 649999.00, 'Electronics', 50, 'High-performance laptop with Intel Core i7'),
                ('Wireless Mouse', 14999.00, 'Accessories', 200, 'Ergonomic wireless mouse with long battery life'),
                ('USB-C Hub', 24999.00, 'Accessories', 150, '7-in-1 USB-C hub with HDMI and SD card reader'),
                ('Monitor 4K 27"', 249999.00, 'Electronics', 30, '27-inch 4K IPS monitor with HDR support'),
                ('Mechanical Keyboard', 44999.00, 'Accessories', 100, 'Compact mechanical keyboard with RGB lighting');

                INSERT INTO transactions (user_id, product_id, amount, status) VALUES
                (1, 1, 1299.99, 'completed'),
                (1, 2, 29.99, 'completed'),
                (2, 3, 49.99, 'completed'),
                (2, 4, 499.99, 'pending'),
                (3, 5, 89.99, 'completed'),
                (4, 1, 1299.99, 'refunded'),
                (4, 2, 29.99, 'completed'),
                (5, 3, 49.99, 'completed'),
                (5, 4, 499.99, 'completed'),
                (3, 1, 1299.99, 'pending');
            """)
        conn.commit()
        log.info("Database seeded successfully.")
