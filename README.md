<div style="display: flex; align-items: center; justify-content: center; text-align: center;">
  <img src="https://coursereport-s3-production.global.ssl.fastly.net/uploads/school/logo/219/original/CT_LOGO_NEW.jpg" width="100" style="margin-right: 10px;">
  <div>
    <h1><b>📖 Capstone Project</b></h1>
  </div>
</div>

## Introduction
Welcome to this hands-on mini-project where you'll be creating a Streamlit web application to showcase EDA using supermarket data! In this project I went through and analyzed a data set from Kaggle that you can find here:  [https://www.kaggle.com/code/fareedalianwar/supermarket-sales](url)


**Context**
The growth of supermarkets in most populated cities are increasing and market competitions are also high. The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data. Predictive data analytics methods are easy to apply with this dataset.

**Attribute information**
Invoice id: Computer generated sales slip invoice identification number
Branch: Branch of supercenter (3 branches are available identified by A, B and C).
City: Location of supercenters
Customer type: Type of customers, recorded by Members for customers using member card and Normal for without member card.
Gender: Gender type of customer
Product line: General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel
Unit price: Price of each product in $
Quantity: Number of products purchased by customer
Tax: 5% tax fee for customer buying
Total: Total price including tax
Date: Date of purchase (Record available from January 2019 to March 2019)
Time: Purchase time (10am to 9pm)
Payment: Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)
COGS: Cost of goods sold
Gross margin percentage: Gross margin percentage
Gross income: Gross income
Rating: Customer stratification rating on their overall shopping experience (On a scale of 1 to 1)

Encoded Data used for model: 

Encoding column: Branch
Original values: ['A' 'C' 'B']
Mapping:
  A → 0
  B → 1
  C → 2

Encoding column: City
Original values: ['Yangon' 'Naypyitaw' 'Mandalay']
Mapping:
  Mandalay → 0
  Naypyitaw → 1
  Yangon → 2

Encoding column: Customer type
Original values: ['Member' 'Normal']
Mapping:
  Member → 0
  Normal → 1

Encoding column: Gender
Original values: ['Female' 'Male']
Mapping:
  Female → 0
  Male → 1

Encoding column: Product line
Original values: ['Health and beauty' 'Electronic accessories' 'Home and lifestyle'
 'Sports and travel' 'Food and beverages' 'Fashion accessories']
Mapping:
  Electronic accessories → 0
  Fashion accessories → 1
  Food and beverages → 2
  Health and beauty → 3
  Home and lifestyle → 4
  Sports and travel → 5

Encoding column: Payment
Original values: ['Ewallet' 'Cash' 'Credit card']
Mapping:
  Cash → 0
  Credit card → 1
  Ewallet → 2

**Conclusion **
Through data exploration I began to realize there was something fishy about the data. The same gender always went to the same branches as an example. There also was very low correlation levels between the columns. The dataset also had a lot of unneeded noise I handled with rounding. 

I would go back to the owner of the supermarkets and suggest that they check on how the data is collected to make sure there aren't problems in the collection methods. 

I would also note if they wanted to drive traffic to certain branches, they could look at how targetting certain customer types, payment types and genders could have an impact on improving flow into any given specific branch. 

