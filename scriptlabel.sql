SELECT a.*, (summation*1.311/11.5)/2 AS rank FROM 
(SELECT b.*, ((num_critic_for_reviews*10/(SELECT MAX(num_critic_for_reviews) FROM test)) +
((revenue-budget)*10*2/(SELECT MAX(revenue-budget) FROM test)) +
(num_voted_users*10/(SELECT MAX(num_voted_users) FROM test)) +
(cast_total_facebook_likes*10*0.5/(SELECT MAX(cast_total_facebook_likes) FROM test)) +
(num_user_for_reviews*10/(SELECT MAX(num_user_for_reviews) FROM test)) +
(imdb_score*10*2/(SELECT MAX(imdb_score) FROM test)) +
(movie_facebook_likes*10*0.5/(SELECT MAX(movie_facebook_likes) FROM test)) +
(vote_average*10*2/(SELECT MAX(vote_average) FROM test)) +
(vote_count*10/(SELECT MAX(vote_count) FROM test)) +
(cast_power*10*0.5/(SELECT MAX(cast_power) FROM test))) AS summation
FROM test AS b)a
#-----------------------------------------------------------------------------

SELECT a.*, (summation*1.311/11.5)/2 AS rank,
CASE
WHEN ((summation*1.311/11.5)/2 >=1.3 && (summation*1.311/11.5)/2 <2.1) THEN "1"
WHEN ((summation*1.311/11.5)/2 >=2.1 && (summation*1.311/11.5)/2 <3.1) THEN "2"
WHEN ((summation*1.311/11.5)/2 >=3.1 && (summation*1.311/11.5)/2 <4.0) THEN "3"
WHEN ((summation*1.311/11.5)/2 >=4.0)  THEN "4"
WHEN ((summation*1.311/11.5)/2 >= 0 && (summation*1.311/11.5)/2 < 1.3) THEN "0"
END AS Rank_Modified
FROM 
(SELECT b.*, ((num_critic_for_reviews*10/(SELECT MAX(num_critic_for_reviews) FROM test)) +
((revenue-budget)*10*2/(SELECT MAX(revenue-budget) FROM test)) +
(num_voted_users*10/(SELECT MAX(num_voted_users) FROM test)) +
(cast_total_facebook_likes*10*0.5/(SELECT MAX(cast_total_facebook_likes) FROM test)) +
(num_user_for_reviews*10/(SELECT MAX(num_user_for_reviews) FROM test)) +
(imdb_score*10*2/(SELECT MAX(imdb_score) FROM test)) +
(movie_facebook_likes*10*0.5/(SELECT MAX(movie_facebook_likes) FROM test)) +
(vote_average*10*2/(SELECT MAX(vote_average) FROM test)) +
(vote_count*10/(SELECT MAX(vote_count) FROM test)) +
(cast_power*10*0.5/(SELECT MAX(cast_power) FROM test))) AS summation
FROM test AS b)a
#-------------------------------------------------------------------------------------------

SELECT ft.*,(FLOOR(GREATEST(r12,r23,r13)/2)-1) AS label FROM 
(SELECT t.*,t.ranking1+t.ranking2 AS r12, t.ranking2 + ranking3 AS r23 ,
t.ranking1+t.ranking3 AS r13, (t.revenue-t.budget) AS profit FROM 
(SELECT *, 
CASE 
WHEN (revenue-budget)< 0.5*1000000 THEN "1"
WHEN (revenue-budget)>= 0.5*1000000 && (revenue-budget)< 1*1000000 THEN "2"
WHEN (revenue-budget)>=1*1000000 && (revenue-budget)< 40*1000000 THEN "3"
WHEN (revenue-budget)>=40*1000000 && (revenue-budget)< 150*1000000 THEN "4"
WHEN (revenue-budget)>= 150 *1000000 THEN "5"
END AS ranking1,
CASE 
WHEN (vote_average)< 4.0 THEN "1"
WHEN (vote_average)>= 4.0 && (vote_average)<5.5  THEN "2"
WHEN (vote_average)>=5.5 && (vote_average)< 6.5  THEN "3"
WHEN (vote_average)>=6.5 && (vote_average)< 7.8 THEN "4"
WHEN (vote_average)>= 7.8 THEN "5"
END AS ranking2,

CASE 
WHEN (num_critic_for_reviews)< 200 THEN "1"
WHEN (num_critic_for_reviews)>= 200 && (num_critic_for_reviews)<500  THEN "2"
WHEN (num_critic_for_reviews)>= 500 && (num_critic_for_reviews)< 600  THEN "3"
WHEN (num_critic_for_reviews)>= 600 && (num_critic_for_reviews)< 700 THEN "4"
WHEN (num_critic_for_reviews)>= 700 THEN "5"
END AS ranking3

FROM test)t)ft
