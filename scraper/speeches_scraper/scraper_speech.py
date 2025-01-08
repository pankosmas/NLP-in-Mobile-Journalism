import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

def scrape_prime_minister_pages():
    base_url = "https://www.primeminister.gr/"
    target_class = "td_module_10 td_module_wrap td-animation-stack"
    results = []
    detailed_results = []

    for year in [2024, 2023, 2022, 2021 ,2020]:
        for i in range(1, 13):
            url = f"{base_url}{year}/{i}"
            print(f"Scraping: {url}")

            while url:
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

                    soup = BeautifulSoup(response.text, "html.parser")
                    divs = soup.find_all("div", class_=target_class)

                    for div in divs:
                        title_tag = div.find("h3", class_="entry-title td-module-title")
                        date_tag = div.find("time", class_="entry-date")

                        if title_tag and date_tag:
                            title = title_tag.get_text(strip=True)
                            post_url = title_tag.find("a")["href"]
                            date = date_tag.get_text(strip=True)

                            print(f"Scraping post: {title}")

                            # Visit the post URL and scrape the whole HTML
                            try:
                                post_response = requests.get(post_url)
                                post_response.raise_for_status()

                                post_soup = BeautifulSoup(post_response.text, "html.parser")
                                article_element = post_soup.find("article")
                                if article_element:
                                    paragraphs = article_element.find_all("p")
                                    article_text = " ".join(p.get_text(strip=True) for p in paragraphs)
                                else:
                                    article_text = ""

                                detailed_results.append({
                                    "title": title,
                                    "url": post_url,
                                    "date": date,
                                    "article_text": article_text
                                })
                                
                            except requests.exceptions.RequestException as e:
                                print(f"Failed to scrape post URL {post_url}: {e}")

                        #  results.append(div)

                    # Locate the "next-page" link within the pagination container
                    pagination_div = soup.find("div", class_=lambda x: x and "page-nav" in x)

                    if pagination_div:
                        next_page = pagination_div.find("a", {"aria-label": "next-page"})

                        if next_page:
                            next_page_url = next_page["href"]
                            print(f"Found next-page link: {next_page_url}")
                            url = next_page_url  # Update URL to the next page
                        else:
                            print("No next-page link found.")
                            url = None  # No "next-page" link found
                    else:
                        print("No pagination container found.")
                        url = None  # No pagination container found

                except requests.exceptions.RequestException as e:
                    print(f"Failed to scrape {url}: {e}")
                    break

    return detailed_results # return results


def store_in_mongodb(data):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["prime_minister"]
    collection = db["articles_speeches"]

    if isinstance(data, list):
        collection.insert_many(data)
    else:
        collection.insert_one(data)

    print("Data successfully stored in MongoDB.")


if __name__ == "__main__":
    scraped_data = scrape_prime_minister_pages()

    # Store the data in MongoDB
    store_in_mongodb(scraped_data)

    # Print the number of elements found and the content
    print(f"Total div elements found: {len(scraped_data)}")
    # for idx, div in enumerate(div_elements[:5]):  # Show the first 5 as a sample
    #     print(f"\nDiv {idx + 1}:\n{div}\n")
