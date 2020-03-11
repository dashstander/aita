import praw
import psaw

class AitaSubmissionSearch:
    subreddit = 'AmITheAsshole'
    default_fields = [
        'title',
        'selftext',
        'id',
        'subreddit_id',
        'score',
        'num_comments',
        'removed_by_category',
        'author',
        'author_fullname',
        'retrieved_on',
        'stickied',
        'locked',
        'num_crossposts',
        'permalink',
        'link_flair_text',
        'created_utc'
    ]

    def __init__(self, credentials, size, fields=None):
        self.ps_api = psaw.PushshiftAPI(max_results_per_request=size)
        self.reddit = praw.Reddit(**credentials)
        if fields is None:
            self.fields = self.default_fields

    def search(self, start_date, end_date, return_batch=False):
        params = {
            "subreddit": self.subreddit,
            "since": start_date,
            "before": end_date,
            "filter": self.fields,
            "not:selftext": "[removed]",
            "return_batch": return_batch
        }

        return self.ps_api.search_submissions(**params)

    def get_comments(self, submission):
        comment_search = psaw.PushshiftAPI(r=self.reddit).search_comments(
            submission_id=submission.id,
            return_batch=True
        )
        for comment in comment_search:
            author = comment.author.name
            if author == 'AutoModerator':
                continue
            yield {
                'comment_author': author,
                'comment_body': comment.body,
                'comment_score': comment.score,
                'comment_created_utc': comment.created_utc,
                'comment_id': comment.id,
                'comment_parent_id': comment.parent_id
            }
