#!/bin/bash

# Gets an authorization token
BASE64CRED=$(cat "./cred.txt")  # Replace this with path to credential string
API_TOKEN=$(curl -X "POST" -H "Authorization: Basic $BASE64CRED" -d grant_type=client_credentials https://accounts.spotify.com/api/token | jq '.access_token')
API_TOKEN="${API_TOKEN%\"}"
API_TOKEN="${API_TOKEN#\"}"

declare -a GENRES=('pop' 'rock' 'country' 'electronic' 'indie')
IFS=","

# For each genre
for i in ${GENRES[@]}; do
    # Get 100 tracks in genre
    TRACKS=$(curl "https://api.spotify.com/v1/recommendations?seed_genres=${GENRES[i]}&limit=100" -H "Authorization: Bearer $API_TOKEN" | jq '[.tracks[].id]')
    # For each track
    for j in $TRACKS; do
        # Get its intended popularity
        CLASS=$(curl -s -H "Authorization: Bearer $API_TOKEN" https://api.spotify.com/v1/tracks/${j//[$'\n\"\ \[\]']} | jq '.popularity')
        # Get its data values and export them as a list fo comma seperated values
        echo "$(curl -s -H "Authorization: Bearer $API_TOKEN" https://api.spotify.com/v1/audio-features/${j//[$'\n\"\ \[\]']} | jq '. | "\(.id), \(.danceability), \(.energy), \(.key), \(.loudness), \(.mode), \(.speechiness), \(.instrumentalness), \(.liveness), \(.valence), \(.tempo), \(.duration_ms), \(.time_signature), "' | sed -e 's/^"//' -e 's/"$//')"$CLASS >> data.csv
    done
done

