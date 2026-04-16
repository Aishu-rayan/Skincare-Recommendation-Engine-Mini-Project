# Sephora Skincare Dataset - Comprehensive Description

## Dataset Overview

The `sephora_select_reviews.db` SQLite database contains comprehensive skincare product data and customer reviews from Sephora, with a strong focus on skincare products.

### Statistics
- **Total Customer Reviews**: 238,929 reviews
- **Total Unique Products**: 8,494 products
- **Database Size**: Comprehensive multi-table relational structure

---

## Database Structure

### Table 1: `select_customer_reviews`
**Purpose**: Individual customer reviews and ratings

**Record Count**: 238,929

**Columns** (23 fields):
| Column | Type | Description |
|--------|------|-------------|
| `primary_category` | TEXT | Main product category (e.g., "Skincare") |
| `secondary_category` | TEXT | Sub-category (e.g., "Moisturizers", "Treatments") |
| `tertiary_category` | TEXT | Specific product type (e.g., "Face Serums", "Moisturizers") |
| `author_id` | TEXT | Unique reviewer identifier |
| `brand_id` | TEXT | Brand identifier |
| `brand_name` | TEXT | Brand name |
| `product_id` | TEXT | Product identifier (links to product_info table) |
| `product_name` | TEXT | Full product name |
| `review_title` | TEXT | Customer's review headline |
| `review_text` | TEXT | Full review text (key field for analysis) |
| `rating` | TEXT | Star rating (1-5 scale) |
| `loves_count` | TEXT | Number of users who "loved" the review |
| `is_recommended` | TEXT | Boolean - customer recommends product |
| `helpfulness` | TEXT | Helpfulness score/percentage |
| `total_feedback_count` | TEXT | Total feedback received on review |
| `submission_time` | TEXT | Date/time review was posted |
| `skin_tone` | TEXT | Customer's skin tone (e.g., "light", "medium", "tan", "deep") |
| `eye_color` | TEXT | Customer's eye color |
| `skin_type` | TEXT | Customer's skin type (e.g., "dry", "oily", "normal", "combination") |
| `price_usd` | TEXT | Product price in USD |
| `ingredients` | TEXT | Product ingredient list (JSON/Array format) |
| `price_usd-1` | TEXT | Alternative price field (possible duplicate) |
| `highlights` | TEXT | Product features/highlights (Array format) |

### Table 2: `product_info`
**Purpose**: Product master data and metadata

**Record Count**: 8,494

**Columns** (27 fields):
| Column | Type | Description |
|--------|------|-------------|
| `product_id` | TEXT | Unique product identifier |
| `product_name` | TEXT | Product name |
| `brand_id` | TEXT | Brand identifier |
| `brand_name` | TEXT | Brand name |
| `loves_count` | TEXT | Total "loves" across all reviews |
| `rating` | TEXT | Average product rating |
| `reviews` | TEXT | Number of reviews for product |
| `size` | TEXT | Product size/volume |
| `variation_type` | TEXT | Type of variation (e.g., "Size + Concentration + Formulation") |
| `variation_value` | TEXT | Specific variation value |
| `variation_desc` | TEXT | Variation description |
| `ingredients` | TEXT | Ingredients list |
| `price_usd` | TEXT | Product price in USD |
| `value_price_usd` | TEXT | Value/regular price |
| `sale_price_usd` | TEXT | Sale price (if applicable) |
| `limited_edition` | TEXT | Limited edition flag |
| `new` | TEXT | New product flag |
| `online_only` | TEXT | Online-only flag |
| `out_of_stock` | TEXT | Stock status |
| `sephora_exclusive` | TEXT | Sephora exclusive flag |
| `highlights` | TEXT | Product highlights/benefits |
| `primary_category` | TEXT | Main category |
| `secondary_category` | TEXT | Sub-category |
| `tertiary_category` | TEXT | Specific product category |
| `child_count` | TEXT | Number of variants/children products |
| `child_max_price` | TEXT | Maximum price among variants |
| `child_min_price` | TEXT | Minimum price among variants |

---

## Key Data Insights

### Review Data Characteristics

**Sample Review Example:**
```
Product: Fenty Skin Hydra Vizor Invisible Moisturizer (SPF 30)
Rating: 5 stars
Recommended: Yes

Review Text: "This product is amazing! It smells amazing as well. I've been using it 
for 6 months now and I would highly recommend! [Used] because... I've struggled with 
cystic acne for many years, I needed something that would protect my skin without 
clogging my pores."

Customer Profile:
- Skin Type: Combination
- Skin Tone: Light-Medium
- Eye Color: Hazel

Benefits Mentioned: Non-pore-clogging, acne-safe
```

### Rich Customer Profiling Data
Each review includes:
- **Skin Type** variations: dry, oily, normal, combination, sensitive (implied)
- **Skin Tone**: light, lightMedium, mediumTan, medium, tan, deep
- **Eye Color**: brown, green, hazel, blue (visual profile data)
- **Skin Concerns**: Acne, redness, wrinkles, clogged pores (extracted from reviews)

### Review Quality Metrics
- **Loves Count**: Indicates helpful/popular reviews (0-64,620 range observed)
- **Recommendation Flag**: Boolean indicating customer would recommend
- **Helpfulness Score**: Percentage score of review helpfulness
- **Feedback Count**: Total interactions on the review

### Product Information Richness
- **Ingredient Lists**: Detailed chemical composition in array format
- **Product Highlights**: Pre-tagged benefits (e.g., "Hyaluronic Acid", "Hydrating", "Oil Free")
- **Categorization**: 3-level category hierarchy
- **Pricing Variants**: Multiple price points for different sizes/variations
- **Exclusivity Flags**: Limited edition, Sephora exclusive, online-only status

---

## Data Quality & Opportunities

### Strengths
1. **Large Scale**: ~240K reviews provide statistically significant insights
2. **Rich Customer Context**: Skin type, tone, eye color enable user profiling
3. **Detailed Review Text**: Unstructured data rich with experiential insights
4. **Product Taxonomy**: Well-organized 3-level category structure
5. **Ingredient Information**: Chemical composition available for recommendation
6. **Engagement Metrics**: Multiple quality indicators for reviews

### Data Variations & Considerations
- Some fields have NULL/NaN values (review titles, variations)
- Price fields may have formatting inconsistencies
- Ingredients/highlights stored as string arrays (requires parsing)
- Rating 

is numeric but stored as TEXT
- Date formats appear consistent but need validation

---

## Relevant Information for Recommendation System

### From Review Text (Unstructured)
The review text contains valuable experiential information:
- **Positive Effects**: "glowing skin", "hydrating", "no white cast", "clogged pores-free"
- **Negative Effects**: "broke out", "irritating", "fragrance-sensitive"
- **Texture/Feel**: "lightweight", "greasy", "buttery feel", "dewy"
- **Performance**: "works well under makeup", "long-lasting SPF protection"
- **Ingredient Sensitivities**: "retinol-sensitive", "fragrance-free preference"
- **Skin Condition Benefits**: "reduces redness", "fades acne scars", "brightening"

### From Structured Fields
- **How Product Performs by Skin Type**: Can aggregate reviews by `skin_type` → analyze ratings/recommendations
- **Ingredient Compatibility**: Link `ingredients` + `skin_type` + `rating`
- **Price-Performance**: Correlate `price_usd` with ratings by customer segment
- **Product Category Performance**: Secondary/tertiary categories reveal product use context

---

## Product Category Distribution

**Sample Categories Identified:**
- Skincare
  - Moisturizers
  - Treatments (Face Serums, Masks)
  - Cleansers
  - Sunscreen/SPF Products
  - Fragrance (Non-skincare, but present in data)

---

## Potential for LLM Analysis

The dataset is ideal for:

1. **Effect Extraction** (as per Glowe implementation):
   - Parse review text with LLM to extract:
     - Positive effects: `["Hydration", "Reduced redness", "No white cast"]`
     - Negative effects: `["Breakouts", "Strong fragrance"]`
     - Texture experience: `["Lightweight", "Creamy"]`
     - Ingredient interactions: `["Works well with retinol", "Avoid with acids"]`

2. **User Preference Mapping**:
   - Skin type + effects → Product recommendations
   - Skin concerns + positive effects → Matching products
   - Ingredient sensitivities → Exclusion lists

3. **Routine Building**:
   - Category segmentation (morning/evening use)
   - Product compatibility (ingredient analysis)
   - Sequencing logic (pH, active ingredients timing)

---

## Next Steps for Your Analysis

1. **Extract Effects from Reviews**: Run LLM analysis on review_text for each review
2. **Aggregate by Product**: Summarize positive/negative effects per product_id
3. **Create Enhanced Product Records**: Store extracted effects + original product_info
4. **Embed All Features**: TF-IDF on review text, embeddings for effect descriptions
5. **Build Recommendation Engine**: Match user profile → effects they need → compatible products

