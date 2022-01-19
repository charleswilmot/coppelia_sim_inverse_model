

if __name__ == '__main__':
    from database import Database
    db_1 = Database('../databases/thesis_experiments.db')
    db_2 = Database('../databases/thesis_experiments_v2.db')
    db_2_groups = db_2.group_by_matching_experiments()
    for group in db_2_groups:
        matchings = db_2.get_matching_experiments_ids(db_1, group[0])
        print(f"db_2's group {group} matches with {matchings} in db_1")


    for group in db_2_groups:
        if len(group) > 1:
            print(group)
            for experiment_id in group:
                df = db_2.get_dataframe(f"SELECT collection FROM experiments WHERE experiment_id='{experiment_id}'")
                collection = df.collection.values[0]
                print(experiment_id, collection)
