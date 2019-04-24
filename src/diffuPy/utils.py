import itertools

def print_dict_dimentions(entites_db, title):
    total = 0
    print(title)
    for k1, v1  in entites_db.items():
        m = ''
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                m += f'{k2}({len(v2)}), '
                total += len(v2)
        else:
            m += f'{len(v1)} '
            total += len(v1)

        print(f'Total number of {k1}: {m} ')

    print(f'Total: {total} ')

def get_labels_set_from_dict(entities):
    if isinstance(list(entities.values())[0], dict):
        return set(itertools.chain.from_iterable(itertools.chain.from_iterable(entities.values())))
    else:
        return set(itertools.chain.from_iterable(entities.values()))

def check_substrings(dataset_nodes, db_nodes):
    mapping_substrings = set()

    for entity in dataset_nodes:
        if isinstance(entity, tuple):
            for subentity in entity:
                for entity_db in db_nodes:
                    if isinstance(entity_db, tuple):
                        for subentity_db in entity_db:
                            if subentity_db in subentity or subentity in subentity_db:
                                mapping_substrings.add(subentity_db)
                                break
                        break
                    else:
                        if entity_db in subentity or subentity in entity_db:
                            mapping_substrings.add(entity_db)
                            break
        else:
            for entity_db in db_nodes:
                if isinstance(entity_db, tuple):
                    for subentity_db in entity_db:
                        if subentity_db in subentity or subentity in subentity_db:
                            mapping_substrings.add(subentity_db)
                            break
                    break
                else:
                    if entity_db in entity or entity in entity_db:
                        mapping_substrings.add(entity_db)
                        break

    return mapping_substrings


