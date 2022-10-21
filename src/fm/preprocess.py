import json
import logging
import os
import pathlib
import shutil
import requests

import numpy as np
import polars as pl
import tensorflow as tf

from typing import Optional, Dict, Union, Tuple

ML_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
BATCH_SIZE = 1_000_000
TRAIN_DATA_DIR = "train"
VALIDATION_DATA_DIR = "validation"
MOVIE_INDEX_MAP_FILE_NAME = "movie_index_map.json"
GENRE_INDEX_MAP_FILE_NAME = "genre_index_map.json"


def preprocess(local_directory: pathlib.Path, logger: logging.Logger):
    logger.info("Download and unpack movielens(25 million) data")
    unpacked_archive_path = download_from_url(ML_25M_URL, local_directory)
    archive_path = unpack_archive(unpacked_archive_path)

    logger.info("Load movie metadata")
    movies = load_movie_metadata(archive_path)

    logger.info("Generate index maps on movie, genre from metadata")
    movie_index_map = define_metadata_index_map(movies, "movieId")
    genre_index_map = define_metadata_index_map(movies, "genres")

    logger.info("Convert genre names in metadata into padded list of indices")
    movies = convert_movie_metadata(movies, movie_index_map, genre_index_map)
    movies = pad_zeros_to_genre(movies)

    logger.info("Join metadata into user ratings data")
    ratings = construct_ratings_data(movies, movie_index_map, archive_path)

    logger.info("Split ratings data and save each data as TFRecord file")
    train_data, validation_data = split_ratings(ratings)
    save_data_as_tfrecords(train_data, TRAIN_DATA_DIR, local_directory, logger)
    save_data_as_tfrecords(validation_data, VALIDATION_DATA_DIR, local_directory, logger)

    logger.info("Save index maps as json format")
    save_index_map(movie_index_map, MOVIE_INDEX_MAP_FILE_NAME, local_directory)
    save_index_map(genre_index_map, GENRE_INDEX_MAP_FILE_NAME, local_directory)


def download_from_url(
    url: str, local_directory: Optional[pathlib.Path] = None
) -> pathlib.Path:
    """
    Download file from submitted url which ends with file name to be downloaded

    Args:
        url: url to download file
        local_directory: path to local directory to download file

    Returns:
        pathlib.Path that contains full file path of downloaded file
    """
    file_name = url.split("/")[-1]
    if local_directory is None:
        local_directory = pathlib.Path(os.environ.get("PWD"))
    local_directory.mkdir(parents=True, exist_ok=True)
    file_path = local_directory.joinpath(file_name)

    if not file_path.exists():
        with requests.get(url, stream=True) as request:
            with open(file_path, "wb") as file:
                shutil.copyfileobj(request.raw, file)
    return file_path


def unpack_archive(archive_path: pathlib.Path) -> pathlib.Path:
    """
    Unpack archived file and return directory of unpacked archive

    Args:
        archive_path: pathlib.Path that contains full file path of archive to be unpacked

    Returns:
        pathlib.Path that contains full file path of unpacked archive folder
    """
    archive_directory = archive_path.parent
    shutil.unpack_archive(archive_path, archive_directory)
    unpacked_archive_path = str(archive_path).split(".")[0]
    return pathlib.Path(unpacked_archive_path)


def load_movie_metadata(archive_path: pathlib.Path) -> pl.DataFrame:
    """
    Load metadata that contains movieId and corresponding genre information

    Args:
        archive_path: pathlib.Path of unpacked movielens archive

    Returns:
        dataframe that contains genre information of each movie
    """
    return (
        pl.read_csv(archive_path / "movies.csv")
        .select(["movieId", "genres"])
        .with_column(pl.col("genres").str.split("|"))
        .explode("genres")
    )


def define_metadata_index_map(
    movies: pl.DataFrame, column_name: str
) -> Dict[Union[str, int], int]:
    """
    Extract unique items of a submitted column in metadata and convert them into index

    Args:
        movies: dataframe that contains genre information of each movie
        column_name: name of column to map each value of item into index

    Returns:
        dict from item to corresponding index
    """
    idx2item = dict(enumerate(movies[column_name].unique(), start=1))  # 0 is reserved for padding index
    return dict([(item, idx) for idx, item in idx2item.items()])


def convert_movie_metadata(
    movies: pl.DataFrame, movie_index_map: Dict[int, int], genre_index_map: Dict[str, int]
) -> pl.DataFrame:
    """
    Convert movieId into movie index and genre names into padded list of genre indices

    Args:
        movies: dataframe that contains genre information of each movie as name
        movie_index_map: dictionary that maps each movieId into corresponding index
        genre_index_map: dictionary that maps each genre name into corresponding index

    Returns:
        dataframe of movie index and list genre indices
    """
    return (
        movies
        .with_columns([
            (pl.col("movieId").apply(lambda x: movie_index_map[x])),
            (pl.col("genres").apply(lambda x: genre_index_map[x]))
        ])
        .groupby("movieId")
        .agg_list()
    )


def pad_zeros_to_genre(movies: pl.DataFrame):
    """
    Pad zeros to genres to equalize variable length of genre index list

    Args:
        movies: dataframe of movie index and list genre indices

    Returns:
        dataframe of movie index and list of zero-padded genre indices
    """
    pad_size = movies.select(pl.col("genres").arr.lengths()).max().to_dicts()[0]["genres"]
    num_movies = len(movies)
    movie_ids = np.arange(1, num_movies + 1).repeat(pad_size)
    padding_data = pl.DataFrame({
        "movieId": movie_ids,
        "padding": np.zeros(len(movie_ids), dtype=int)
    }).groupby("movieId").agg_list()
    return (
        movies.join(padding_data, on="movieId").select([
            "movieId", pl.col("genres").arr.concat("padding").arr.head(pad_size)
        ])
    )


def construct_ratings_data(
    movies: pl.DataFrame,
    movie_index_map: Dict[int, int],
    archive_path: pathlib.Path
) -> pl.DataFrame:
    """
    Attach list of genre indices on ratings matrix

    Args:
        movies: metadata of each movie containing its genre information of index list
        movie_index_map: index map from movieId to corresponding index
        archive_path: pathlib.Path of unpacked movielens archive

    Returns:
        user rating data with movie index and corresponding list of padded genre indices
    """
    return (
        pl.read_csv(archive_path / "ratings.csv")
        .select(["userId", "movieId", "rating"])
        .with_column(pl.col("movieId").apply(lambda x: movie_index_map[x]))
        .join(movies, on="movieId", how="inner")
    )


def split_ratings(ratings: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    split ratings data into two partitions which represents train, validation data respectively

    Args:
        ratings: user rating data with movie index and corresponding list of padded genre indices

    Returns:
        two partitions of ratings data
    """
    validation_data = ratings.groupby("userId").tail(5)
    train_data = ratings.join(validation_data, on=["userId", "movieId"], how="anti")
    return train_data, validation_data


def save_data_as_tfrecords(
    ratings: pl.DataFrame,
    directory_name: str,
    local_directory: pathlib.Path,
    logger: logging.Logger
) -> None:
    """
    convert and save submitted ratings partition into TFRecord format

    Args:
        ratings: user rating data with movie index and corresponding list of padded genre indices
        directory_name: name of data directory to save partitions of submitted ratings data
        local_directory: root directory of data directory
        logger: logger to print status of converting process
    """
    data_directory = local_directory.joinpath(directory_name)
    data_directory.mkdir(exist_ok=True, parents=True)
    num_partition = len(ratings) // BATCH_SIZE + 1
    for index, pointer in enumerate(range(0, len(ratings), BATCH_SIZE), start=1):
        logger.info(f"Fill {directory_name} ({index}/{num_partition})")
        ratings_partition = ratings[pointer:(pointer + BATCH_SIZE)]
        file_path = data_directory.joinpath(f"partition_{index:02}.tfrecord")
        _convert_partition_into_tfrecord_file(ratings_partition, str(file_path))


def _convert_partition_into_tfrecord_file(ratings_partition: pl.DataFrame, file_path: str):
    with tf.io.TFRecordWriter(file_path) as writer:
        for rating in ratings_partition.to_dicts():
            feature = {
                "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[rating["userId"]])),
                "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[rating["movieId"]])),
                "genres": tf.train.Feature(int64_list=tf.train.Int64List(value=rating["genres"])),
                "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[rating["rating"]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)


def save_index_map(
    index_map: Dict[Union[str, int], int],
    file_name: str,
    local_directory: pathlib.Path
) -> None:
    """
    save index map into JSON format

    Args:
        index_map: movie_index_map or genre_index_map
        file_name: name of JSON file
        local_directory: directory to save JSON file
    """
    with open(local_directory.joinpath(file_name), "w") as file:
        json.dump(index_map, file)
