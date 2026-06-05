from datetime import timedelta
from feast import Entity, ValueType

user_entity = Entity(
    name="user", 
    join_keys=["user_id"], 
    description="Unique identifier for a user",
    value_type=ValueType.INT64
)

movie_entity = Entity(
    name="movie", 
    join_keys=["movie_id"], 
    description="Unique identifier for a movie",
    value_type=ValueType.INT64
)