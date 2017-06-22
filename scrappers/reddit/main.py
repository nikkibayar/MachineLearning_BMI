import praw
import os
import json
import datetime
import mimetypes
import re


'''
Follow instructions here on how to get your credentials:
https://github.com/reddit/reddit/wiki/OAuth2 '''

OUTPUT_FOLDER="json_out"
SUBREDDIT="progresspics"
username = os.environ['REDDIT_USERNAME']
user_agent = "BMI/0.1 by " + username
client_id = os.environ['REDDIT_CLIENT_ID']
client_secret = os.environ['REDDIT_CLIENT_SECRET']

def get_redit():
    return praw.Reddit( client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent)

def get_weight(title):
    weight_info = title[title.index("[")+1:title.index("]")]
    weight_info = re.sub(r'\s+', '',weight_info)
    weight_info = re.split('<|>', weight_info )

    weight = {
        'before_weight': weight_info[0],
        'after_weight': weight_info[1].split("=")[0],
    }
    return weight

def get_gender_and_height(submission):
    return submission.link_flair_text.split("(")[0].strip().split(" ")

def get_object(submission):
    gender_and_height = get_gender_and_height(submission)
    object = {
                "title" : submission.title,
                "url" : submission.url,
                "id" : submission.id,
                "created_utc": datetime.datetime.fromtimestamp(
                    int(submission.created_utc)
                ).strftime('%Y-%m-%d %H:%M:%S'),
                "height": gender_and_height[1],
                "gender": gender_and_height[0]
            }
    object.update( get_weight(submission.title) )
    return object

def is_url_image(url):
    mimetype,encoding = mimetypes.guess_type(url)
    return (mimetype and mimetype.startswith('image'))

def load_previous_submissions():
    if os.path.exists(OUTPUT_FOLDER) is False:
        return set()

    previous_reddit_ids = set()
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".json") :
            with open(os.path.join(OUTPUT_FOLDER, filename)) as data_file:
                data = json.load(data_file)
                previous_reddit_ids.update( [ x['id'] for x in data ] )


    return previous_reddit_ids


def get_new_submissions():
    print("Loading cached submissions in {}".format(OUTPUT_FOLDER))
    cached_posts = load_previous_submissions()
    print("Loaded {} cached submissions".format(len(cached_posts)))
    parsing_errors_count = 0
    preexisting_posts = 0
    not_an_image_posts = 0

    reddit = get_redit()

    new_submissions = []
    for submission in reddit.subreddit(SUBREDDIT).submissions(start=1451606400, end=1483228800):
        if str(submission) in cached_posts:
            preexisting_posts+=1
            continue
        try:
            object = get_object(submission)
            if is_url_image(object['url']):
                new_submissions.append( object )
            else:
                not_an_image_posts+=1
        except:
            parsing_errors_count+=1

    if len(new_submissions) > 0:
        if os.path.exists(OUTPUT_FOLDER) is False:
            os.makedirs(OUTPUT_FOLDER)

        output_file = os.path.join(OUTPUT_FOLDER, str(datetime.datetime.now()) + ".json" )

        with open(output_file, 'w') as outfile:
            json.dump(new_submissions, outfile, indent=4, sort_keys=True)

        print("File saved in {} with a total of {} new entries".format(output_file, len(new_submissions)))
        print("Parsing errors: {}".format(parsing_errors_count))
        print("Ignored posts (already in existing cache): {}".format(preexisting_posts))
        print("Ignored posts (doesn't contain an image): {}".format(not_an_image_posts))
    else:
        print("No new submissions found. Nothing was saved.")


get_new_submissions()

