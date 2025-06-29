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

# Download NLTK resources (run once)
import nltk

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
        """Load and parse JSON data"""
        logger.info(f"Loading data from {filepath}")
        try:
            with open(filepath) as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} job offers")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess_text(self, text):
        """Clean and normalize text"""
        try:
            text = text.lower()
            text = re.sub(f'[{string.punctuation}]', '', text)
            words = text.split()
            words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return ""

    def combine_fields(self, row):
        """Combine relevant fields into single text"""
        try:
            combined = ' '.join([
                row['Job Offer Title'],
                row['Job Offer Summarize'],
                ' '.join(row['Key Skills Required']),
                ' '.join(row['Essential Qualifications']),
                row['Description']
            ])
            return combined
        except Exception as e:
            logger.warning(f"Error combining fields for row: {str(e)}")
            return ""

    def preprocess_data(self, df):
        """Preprocess the entire dataframe"""
        logger.info("Starting data preprocessing")

        # Add progress bar for combining fields
        tqdm.pandas(desc="Combining fields")
        df['combined_text'] = df.progress_apply(self.combine_fields, axis=1)

        # Add progress bar for text cleaning
        tqdm.pandas(desc="Cleaning text")
        df['cleaned_text'] = df['combined_text'].progress_apply(self.preprocess_text)

        logger.info("Data preprocessing completed")
        return df

    def generate_embeddings(self, texts):
        """Create sentence embeddings"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            logger.info("Embeddings generated successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def cluster_offers(self, embeddings):
        """Cluster offers using DBSCAN"""
        logger.info("Starting clustering with DBSCAN")
        try:
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            logger.info(f"Clustering completed. Found {n_clusters} clusters")
            logger.info(f"Cluster distribution: {pd.Series(clustering.labels_).value_counts().to_dict()}")
            return clustering.labels_
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise

    def extract_keywords(self, texts, n_keywords=10):
        """Extract top keywords using TF-IDF"""
        logger.info(f"Extracting top {n_keywords} keywords from {len(texts)} texts")
        try:
            vectorizer = TfidfVectorizer(max_features=n_keywords)
            X = vectorizer.fit_transform(texts)
            keywords = vectorizer.get_feature_names_out()
            logger.debug(f"Top keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    def get_cluster_summary(self, df, clusters):
        """Generate summary for each cluster"""
        logger.info("Generating cluster summaries")
        df['cluster'] = clusters
        cluster_summary = {}

        # Wrap cluster iteration with progress bar
        for cluster_id in tqdm(df['cluster'].unique(), desc="Processing clusters"):
            cluster_data = df[df['cluster'] == cluster_id]
            keywords = self.extract_keywords(cluster_data['cleaned_text'])

            cluster_summary[cluster_id] = {
                'top_keywords': list(keywords),
                'num_offers': len(cluster_data),
                'example_titles': cluster_data['Job Offer Title'].tolist()[:3]  # Show first 3
            }

        logger.info(f"Generated summaries for {len(cluster_summary)} clusters")
        return cluster_summary

    def visualize_clusters(self, embeddings, clusters):
        """Create 2D visualization using t-SNE"""
        logger.info("Creating t-SNE visualization")
        try:
            perplexity = min(3, len(embeddings) - 1)  # Set perplexity to n_samples - 1 or 3, whichever is smaller
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            logger.info(f"Running t-SNE dimensionality reduction with perplexity={perplexity}...")
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(10, 8))
            scatter = sns.scatterplot(
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

            # Save the visualization
            plot_filename = f"cluster_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename)
            logger.info(f"Cluster visualization saved as {plot_filename}")
            plt.show()

        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise

    def process_job_offers(self, filepath):
        """Complete processing pipeline"""
        logger.info("Starting job offer processing pipeline")
        try:
            # Load and preprocess data
            df = self.load_data(filepath)
            df = self.preprocess_data(df)

            # Generate embeddings and cluster
            embeddings = self.generate_embeddings(df['cleaned_text'].tolist())
            clusters = self.cluster_offers(embeddings)

            # Generate results
            cluster_summary = self.get_cluster_summary(df, clusters)
            job_to_cluster = dict(zip(df['Id'], clusters))

            # Visualization
            self.visualize_clusters(embeddings, clusters)

            logger.info("Processing pipeline completed successfully")

            return {
                'cluster_summary': cluster_summary,
                'job_to_cluster': job_to_cluster,
                'embeddings': embeddings,
                'clusters': clusters
            }
        except Exception as e:
            logger.error(f"Processing pipeline failed: {str(e)}")
            raise


# Example Usage
if __name__ == "__main__":
    try:
        logger.info("Starting job offer clustering script")

        # Initialize clusterer
        clusterer = JobOfferClusterer()

        # Process job offers (replace with your file path)
        results = clusterer.process_job_offers('processed_parse_jobs.json')

        # Print results
        logger.info("Cluster Summary:")
        for cluster_id, summary in results['cluster_summary'].items():
            logger.info(f"\nCluster {cluster_id}:")
            logger.info(f"  Offers: {summary['num_offers']}")
            logger.info(f"  Top Keywords: {', '.join(summary['top_keywords'])}")
            logger.info(f"  Example Titles: {summary['example_titles']}")

        logger.info("\nJob to Cluster Mapping:")
        logger.info(results['job_to_cluster'])

    except Exception as e:
        logger.critical(f"Script failed: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")