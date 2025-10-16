# TE Connectivity - AI Aces Backend: Database Layer

This module is responsible for managing the PostgreSQL database used in the AI model training and image classification pipeline. It handles:

- Database connection
- Table creation
- Insertion of components, dataset versions, and image metadata

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PostgreSQL**
- **psycopg2** for PostgreSQL access
- **FastAPI** for endpoint exposure
- **Modular design** for configuration and queries

---

## 📁 File Structure

```bash
backend/
├── db.py           # Main database handler for connection, table creation, and insertions
└── db_config.py    # Contains credentials and all SQL queries as Python strings
```
## 🗃️ Table Descriptions

### `components`
| Column        | Type   | Description               |
|---------------|--------|---------------------------|
| `id`          | SERIAL | Primary key               |
| `part_number` | TEXT   | Unique part number        |

### datasets
| Column         | Type      | Description                        |
|----------------|-----------|------------------------------------|
| `id`           | SERIAL    | Primary key                        |
| `component_id` | INT       | Foreign key to `components(id)`    |
| `version`      | INT       | Dataset version                    |
| `created_at`   | TIMESTAMP | Timestamp of creation              |

### `images`
| Column        | Type      | Description                         |
|---------------|-----------|-------------------------------------|
| `id`          | SERIAL    | Primary key                         |
| `dataset_id`  | INT       | Foreign key to `datasets(id)`       |
| `image_path`  | TEXT      | Path to the image file              |
| `part_number` | TEXT      | Redundant field for easier queries  |
| `uploaded_at` | TIMESTAMP | Timestamp of image upload           |

---
