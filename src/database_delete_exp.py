

if __name__ == '__main__':
    from database import Database
    # db = Database('../databases/tmp_version_1.db')
    db = Database('../databases/version_1.db')

    # experiment_ids = (220, 221, 223, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242)
    for experiment_id in experiment_ids:
        print('deleting experiment {}'.format(experiment_id))
        db.delete_exp(experiment_id)
