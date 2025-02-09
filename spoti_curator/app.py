import dotenv

from spoti_curator.recommender import do_recommendation

def run():
    dotenv.load_dotenv()
    do_recommendation()


if __name__ == '__main__':
    run()
