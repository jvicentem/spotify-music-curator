def similitude(embed_1, embed_2):
    """
    Given two embeddings, it calculates their similitude.

    Interacts with storage/embeddings.py to calculate the distance.
    """
    pass

def hard_rules():
    """
    Simple rules that makes a song to be included in the final playlists.

    - Artist is included in fav pl are included in each pl (if enabled)
    - Songs with distance = 0 -> song won't be included in final pl.
    """
    pass

def extract_embedding(song_id, song_audio):
    """
    Given a song audio, it extracts an embedding.

    If it is already calculated and stored, it will retrieve it from storage/embeddings.py.

    If not, it will calculate it.
    """
    pass

def do_recommendatio():
    """
    Call each function to generate the curated playlists.
    """

    # get yaml config

    # get n songs config

    # first, let's do the distance config part

    #   extract embeddings from songs in ref pl that are not in embeddings db
    #   now, iterate through each pl
    #       for each song in pl
    #           extract song clip
    #           extract embedding
    #           calculate similitude
    #           save similitude in a dict of type song_id: min distance to any song

    # depending on the similitudes, songs will go to some or other pls (distance range configured)
    # get the top n songs config with the lowest similitude
    # create pls according to their config (call hard_rules to ensure this)
    pass
