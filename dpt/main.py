from dpt.job_pipeline import job_init

world_size = 10
rounds = 100

if __name__ == '__main__':
    job = job_init(world_size)

    for r in range(0, rounds):
        job()