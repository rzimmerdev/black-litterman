from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time

# Set up the webdriver, use ubuntu default path

# Set up the driver
def main(name):
    path = './chromedriver'
    options = webdriver.ChromeOptions()
    service = Service(path)
    driver = webdriver.Chrome(options=options, service=service)

    # Navigate to the URL
    base = f"https://br.advfn.com/bolsa-de-valores/bmf/{name}/historico/mais-dados-historicos"
    current = 0
    last = 4
    Date1 = "17/07/23"
    Date2 = "16/07/24"

    data = []

    while current < last:
        url = "{base}?current={current}&Date1={Date1}&Date2={Date2}"
        print(url)
        driver.get(url)

        # Wait for the page to load completely
        time.sleep(0.1)  # Adjust the sleep time as needed

        # Find the table
        table = driver.find_element(By.CLASS_NAME, "histo-results")

        # Extract table rows
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Initialize a list to hold the data

        # Iterate through the rows and extract data
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells:  # Only process rows with data cells (skip header)
                row_data = [cell.text for cell in cells]
                data.append(row_data)

        # Print the extracted data
        for row_data in data:
            print(row_data)

        # Increment the current page number
        current += 1

    # Close the driver
    driver.quit()
    return data


# save data to csv
if __name__ == "__main__":
    name = input("Enter the security name: ").replace("'", "").replace('"', "")
    data = main(name)
    with open(f"{name}.csv", "w") as f:
        for row in data:
            f.write(",".join(row) + "\n")
    print("File saved.")
