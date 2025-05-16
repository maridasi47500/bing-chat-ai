import sqlite3

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# Create 'job' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS job (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT NOT NULL
    )
''')

# Create 'sport' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sport (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT NOT NULL
    )
''')

# Create 'arts' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS arts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT NOT NULL
    )
''')

# Optional: Insert some sample data
cursor.execute('INSERT INTO job (name, description) VALUES (?, ?)', ('Software Engineer', 'Develops software applications.'))
cursor.execute('INSERT INTO sport (name, description) VALUES (?, ?)', ('Football', 'A team sport played with a spherical ball.'))
cursor.execute('INSERT INTO arts (name, description) VALUES (?, ?)', ('Painting', 'Creating artwork using paints.'))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and tables with name and description created successfully with sample data!")
