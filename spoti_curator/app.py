import dotenv

from spoti_curator.recommender import do_recommendation

def run():
    dotenv.load_dotenv()
    do_recommendation()


if __name__ == '__main__':
    run()

# simil_df_aux['1_simil_ref'].quantile([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0])
# 0.00    0.388879
# 0.01    0.614830
# 0.10    0.895248
# 0.25    0.954211
# 0.50    0.978941
# 0.75    0.989172
# 0.99    0.997409
# 1.00    0.999477
# Name: 1_simil_ref, dtype: float64