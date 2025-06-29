import json
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
from datetime import datetime

import nltk
from collections import Counter
import ast

nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'job_clustering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JobOfferClusterer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("JobOfferClusterer initialized with embedding model")

    def load_data(self, filepath):
        logger.info(f"Loading data from {filepath}")
        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} job offers")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def get_top_skills_by_cluster(self, df, cluster_id, top_n=15):
        logger.info(f"Extracting top {top_n} skills from Cluster {cluster_id}")
        try:
            cluster_df = df[df['cluster'] == cluster_id].copy()

            # Extract skills from Description or SearchText
            all_words = []
            for text in cluster_df['Description'].fillna('').astype(str) + " " + cluster_df['SearchText'].fillna(
                    '').astype(str):
                words = re.findall(r'\b\w+\b', text.lower())
                words = [w for w in words if w not in self.stop_words and len(w) > 2]
                all_words.extend(words)

            word_counts = Counter(all_words)
            top_skills = word_counts.most_common(top_n)

            logger.info(f"Top skills in cluster {cluster_id}: {top_skills}")
            return top_skills
        except Exception as e:
            logger.error(f"Failed to extract top skills: {str(e)}")
            return []

    def preprocess_text(self, text):
        try:
            # Keep more meaningful terms
            text = text.lower()
            text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove very short words
            text = re.sub(f'[{string.punctuation}]', ' ', text)  # Keep as space
            words = text.split()
            words = [self.lemmatizer.lemmatize(w) for w in words
                     if w not in self.stop_words and not w.isnumeric()]
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return ""

    def combine_fields(self, row):
        try:
            combined = ' '.join([
                str(row.get('JobOfferTitle', '')),
                str(row.get('Description', '')),
                str(row.get('CompanyName', '')),
                str(row.get('SearchText', ''))
            ])
            return combined
        except Exception as e:
            logger.warning(f"Error combining fields: {str(e)}")
            return ""

    def preprocess_data(self, df):
        logger.info("Starting data preprocessing")
        tqdm.pandas(desc="Combining fields")
        df['combined_text'] = df.progress_apply(self.combine_fields, axis=1)
        tqdm.pandas(desc="Cleaning text")
        df['cleaned_text'] = df['combined_text'].progress_apply(self.preprocess_text)
        logger.info("Data preprocessing completed")
        return df

    def generate_embeddings(self, texts):
        logger.info(f"Generating embeddings for {len(texts)} texts")
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def cluster_offers(self, embeddings):
        logger.info("Starting clustering with DBSCAN")
        try:
            # More strict parameters
            clustering = DBSCAN(eps=0.3, min_samples=5, metric='cosine').fit(embeddings)
            logger.info(f"Found {len(set(clustering.labels_))} clusters")
            return clustering.labels_
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise

    def extract_keywords(self, texts, n_keywords=10):
        logger.info(f"Extracting top {n_keywords} keywords")
        try:
            vectorizer = TfidfVectorizer(max_features=n_keywords)
            X = vectorizer.fit_transform(texts)
            return vectorizer.get_feature_names_out()
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    def get_cluster_summary(self, df, clusters):
        logger.info("Generating cluster summaries")
        df['cluster'] = clusters
        summary = {}

        for cluster_id in tqdm(df['cluster'].unique(), desc="Processing clusters"):
            cluster_data = df[df['cluster'] == cluster_id]
            keywords = self.extract_keywords(cluster_data['cleaned_text'])

            summary[cluster_id] = {
                'top_keywords': list(keywords),
                'num_offers': len(cluster_data),
                'example_titles': cluster_data['JobOfferTitle'].tolist()[:3]
            }
        return summary

    def visualize_clusters(self, embeddings, clusters):
        logger.info("Creating t-SNE visualization")
        try:
            perplexity = min(30, len(embeddings) - 1)  # Increased from 3 to 30 for better visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                hue=clusters,
                palette='viridis',
                s=100,
                alpha=0.8
            )
            plt.title('Job Offer Clusters (t-SNE projection)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend(title='Cluster ID')

            plot_filename = f"cluster_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"Saved cluster visualization to {plot_filename}")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise

    def process_job_offers(self, filepath):
        logger.info("Starting job offer processing pipeline")
        try:
            df = self.load_data(filepath)
            df = self.preprocess_data(df)
            embeddings = self.generate_embeddings(df['cleaned_text'].tolist())
            clusters = self.cluster_offers(embeddings)
            cluster_summary = self.get_cluster_summary(df, clusters)
            job_to_cluster = dict(zip(df['ID'], clusters))
            self.visualize_clusters(embeddings, clusters)

            return {
                'cluster_summary': cluster_summary,
                'job_to_cluster': job_to_cluster,
                'embeddings': embeddings,
                'clusters': clusters,
                'df': df
            }
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        logger.info("Running clustering script")
        clusterer = JobOfferClusterer()
        results = clusterer.process_job_offers('processed_parse_jobs.json')

        df = results['df']
        cluster_1_offers = df[df['cluster'] == 1]

        # Export Cluster 1
        cluster_1_offers.to_csv("cluster_1_offers.csv", index=False)
        cluster_1_offers.to_json("cluster_1_offers.json", orient='records', indent=4)
        logger.info(f"Exported {len(cluster_1_offers)} offers from Cluster 1 to CSV and JSON.")

        # Get and display top skills from Cluster 1
        top_skills = clusterer.get_top_skills_by_cluster(df, cluster_id=1, top_n=15)
        print("\n🔥 Top Skills in Cluster 1:")
        for skill, count in top_skills:
            print(f"✅ {skill} ({count} times)")

        # Print cluster summaries
        logger.info("\nCluster Summary:")
        for cluster_id, summary in results['cluster_summary'].items():
            logger.info(f"\nCluster {cluster_id}:")
            logger.info(f"  Offers: {summary['num_offers']}")
            logger.info(f"  Top Keywords: {', '.join(summary['top_keywords'])}")
            logger.info(f"  Example Titles: {summary['example_titles']}")

        # Optional: Print each offer in cluster 1
        print("\nSample Offers from Cluster 1:")
        for _, row in cluster_1_offers.head(3).iterrows():
            print(f"\n🔹 ID: {row['ID']}")
            print(f"📌 Title: {row['JobOfferTitle']}")
            print(f"🏢 Company: {row['CompanyName']}")
            print(f"💰 Salary: {row.get('SalaryOrBudgetOffered', 'Not specified')}")
            print(f"📝 Description: {row['Description'][:200]}...")

    except Exception as e:
        logger.critical(f"Script failed: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")