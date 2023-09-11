try:
    from housing_price import ingest_data, score, train

    print("Installation Successfull.")
except Exception as e:
    print("Installition unsuccessful.")
    print(e)
