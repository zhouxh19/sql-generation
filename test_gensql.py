import base

def main():
    db, cursor = base.connect_server("tpch", "postgresql")

if __name__ == '__main__':
    main()