# AWS Deployment Guide — Skincare Recommender

This guide walks you from a fresh AWS student account to a live Streamlit
dashboard backed by an RDS PostgreSQL lookup table. It is written for
someone who has never used AWS before — follow it top to bottom.

---

## 0. Before you start

### Important: student account caveats
AWS Academy / student accounts are usually **restricted**:
- **SageMaker** is often disabled or quota-limited. If so, we run the
  TF-IDF training on **EC2** instead (cheaper and simpler anyway).
- Sessions typically expire every 3–4 hours. Log back in and your
  resources persist, but **stop EC2/RDS when not in use** to preserve
  your lab credit.
- Always check your credit balance in the AWS Academy portal.

### What you need
- AWS student account login (the "Learner Lab" or a normal IAM user)
- The AWS Console URL your professor provided
- This project running locally (Phase 1 + 2 already done)

### Services we will use
| Service | Purpose | Student tier? |
|---|---|---|
| **S3** | Store raw DB, model artifacts, recommendations CSV | ✅ yes |
| **EC2** (t3.micro) | Run TF-IDF training + host Streamlit | ✅ free tier |
| **RDS PostgreSQL** (db.t3.micro) | Serve the recommendations lookup table | ✅ free tier |
| **IAM** | Role that lets EC2 read S3 and reach RDS | ✅ yes |
| SageMaker | Optional — skip on student accounts | ⚠️ often blocked |

**Estimated cost if you leave everything running 24/7:** ~$25/month.
**Cost if you stop EC2+RDS when idle:** ~$1–3/month.

---

## 1. Create an S3 bucket

1. AWS Console → **S3** → *Create bucket*
2. Bucket name: `skincare-recommender-<your-initials>` (must be globally unique)
3. Region: `us-east-1` (pick one and stick to it for every service)
4. Leave all defaults, create.
5. Upload these files under these prefixes:

   ```
   raw/sephora_select_reviews.db          ← the SQLite file from repo root
   ```

   You can drag-and-drop in the S3 console, or from your terminal once
   the AWS CLI is configured (see §2):

   ```bash
   aws s3 cp sephora_select_reviews.db s3://skincare-recommender-<you>/raw/
   ```

The `processed/`, `models/`, `precomputed/`, `logs/` prefixes will be
created automatically when the training script writes to them.

---

## 2. Configure the AWS CLI locally

On your laptop:

```bash
# Windows PowerShell
winget install Amazon.AWSCLI
# Or download the MSI from https://aws.amazon.com/cli/

aws configure
# Paste: AWS Access Key ID, Secret Key, region=us-east-1, output=json
```

Student accounts usually give you **temporary credentials** (Access Key +
Secret + Session Token). If yours does, run:

```bash
aws configure set aws_session_token <TOKEN>
```

Verify: `aws s3 ls` should list your bucket.

---

## 3. Create the RDS PostgreSQL database

1. AWS Console → **RDS** → *Create database*
2. Choose: **Standard create**, engine **PostgreSQL**, version 15.x
3. Templates → **Free tier**
4. Settings:
   - DB instance identifier: `skincare-db`
   - Master username: `skincare`
   - Master password: *(save this — you'll need it as `RDS_PASSWORD`)*
5. Instance config: `db.t3.micro` (free tier)
6. Storage: 20 GB GP2, **uncheck autoscaling**
7. Connectivity:
   - Public access → **Yes** (needed so your laptop and EC2 can reach it)
   - VPC security group → *Create new*, name it `skincare-rds-sg`
8. Additional config → Initial database name: `skincare`
9. Create database. Wait ~5 minutes until status = Available.
10. Click into the DB → copy the **Endpoint** (looks like
    `skincare-db.xxxxx.us-east-1.rds.amazonaws.com`). That is your `RDS_HOST`.
11. Security group → inbound rules → add:
    - Type: PostgreSQL (5432), Source: **My IP** (for your laptop)
    - Type: PostgreSQL (5432), Source: `skincare-ec2-sg` (after §4)

---

## 4. Create the EC2 instance (trainer + dashboard host)

1. AWS Console → **EC2** → *Launch instance*
2. Name: `skincare-app`
3. AMI: **Amazon Linux 2023** (free tier eligible)
4. Instance type: `t3.micro` (free tier) — or `t3.medium` if training is slow
5. Key pair: *Create new*, name `skincare-key`, download the `.pem`
   file and save it somewhere safe
6. Network settings:
   - Create security group `skincare-ec2-sg`
   - Inbound rules:
     - SSH (22) from **My IP**
     - Custom TCP (8501) from **Anywhere** — this is the Streamlit port
7. Storage: 20 GB gp3
8. Advanced → IAM instance profile → *Create new role* (see §5), attach it
9. Launch. Wait until state = running. Copy the **Public IPv4** address.

---

## 5. Create an IAM role for EC2

1. IAM → Roles → *Create role*
2. Trusted entity: **AWS service**, Use case: **EC2**
3. Permissions: attach `AmazonS3FullAccess` (fine for coursework; tighten later)
4. Name: `skincare-ec2-role`
5. Back in EC2 → right-click your instance → Security → *Modify IAM role* → attach `skincare-ec2-role`

Now EC2 can read/write your S3 bucket without any credentials in code.

---

## 6. Provision EC2: Python, repo, and secrets

SSH in from your laptop:

```bash
ssh -i skincare-key.pem ec2-user@<EC2_PUBLIC_IP>
```

On the EC2 box:

```bash
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip git postgresql15
python3.11 -m venv ~/venv
source ~/venv/bin/activate

# Clone your repo (push it to GitHub first if you haven't)
git clone https://github.com/<you>/skincare_app.git
cd skincare_app/skincare-recommender
pip install -r requirements.txt

# Create .env from the example
cp .env.example .env
nano .env
# Fill in:
#   S3_BUCKET=skincare-recommender-<you>
#   RDS_HOST=<the endpoint from §3>
#   RDS_PORT=5432
#   RDS_DB=skincare
#   RDS_USER=skincare
#   RDS_PASSWORD=<your password>
```

---

## 7. Run the pipeline on EC2

```bash
cd ~/skincare_app/skincare-recommender
source ~/venv/bin/activate

# Pull raw DB from S3
aws s3 cp s3://skincare-recommender-<you>/raw/sephora_select_reviews.db \
    ../sephora_select_reviews.db

# Run the pipeline — same commands as local
python -m src.processing.aggregate_reviews --env local
python -m src.processing.build_profiles    --env local
python -m src.modeling.fit_tfidf            --env local
python -m src.modeling.precompute_all       --env local

# Push artifacts back to S3 for durability
aws s3 sync ../models/recommender s3://skincare-recommender-<you>/models/
aws s3 cp   data/processed/precomputed/recommendations.csv \
            s3://skincare-recommender-<you>/precomputed/
```

> We use `--env local` here because on EC2, `../sephora_select_reviews.db`
> is a local file path — the only difference from your laptop is that
> the DB came from S3. True `--env aws` (streaming from S3) will be
> added in Phase 5 once the basics work.

---

## 8. Load recommendations into RDS

This step is Phase 3 of the project. Placeholder until `load_rds.py` is written:

```bash
# Create schema
psql -h $RDS_HOST -U $RDS_USER -d $RDS_DB -f src/database/schema.sql

# Load products + recommendations
python -m src.database.load_rds --env local
```

Verify from your laptop:

```bash
psql -h <RDS_HOST> -U skincare -d skincare \
  -c "SELECT COUNT(*) FROM recommendations;"   # expect 2100
```

---

## 9. Launch the Streamlit dashboard

On EC2:

```bash
cd ~/skincare_app/skincare-recommender
source ~/venv/bin/activate
nohup streamlit run src/dashboard/app.py --server.port 8501 \
    --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

Open in your browser: `http://<EC2_PUBLIC_IP>:8501`

To make it persistent across reboots, create a `systemd` service:

```bash
sudo tee /etc/systemd/system/skincare.service > /dev/null <<'EOF'
[Unit]
Description=Skincare Streamlit dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/skincare_app/skincare-recommender
EnvironmentFile=/home/ec2-user/skincare_app/skincare-recommender/.env
ExecStart=/home/ec2-user/venv/bin/streamlit run src/dashboard/app.py \
    --server.port 8501 --server.address 0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now skincare
```

---

## 10. Shutdown checklist (save your credit!)

When you're done for the day:

```bash
# From your laptop
aws ec2 stop-instances   --instance-ids <EC2_INSTANCE_ID>
aws rds  stop-db-instance --db-instance-identifier skincare-db
```

RDS stop only lasts 7 days — after that AWS auto-starts it. Set a calendar reminder.

Resume later with `start-instances` / `start-db-instance`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aws: command not found` | Install AWS CLI (§2) |
| `Unable to connect to RDS` from laptop | Add your current IP to the `skincare-rds-sg` inbound rules |
| Streamlit page won't load | Check EC2 security group allows port 8501 from your IP |
| `AccessDenied` on S3 from EC2 | IAM role not attached — re-do §5 step 5 |
| SageMaker not available | Expected on student accounts — ignore, we use EC2 |
| Training OOMs on t3.micro | Upgrade to t3.medium temporarily, then downgrade |

---

## If this is too much — granting Claude temporary AWS access

If you'd rather I run the deployment from here, you can create a short-lived
IAM user and share the credentials with me in this chat. **Only do this if
your professor allows it** — student accounts sometimes forbid sharing keys.

1. IAM → Users → *Create user* → name `claude-deployer`
2. Attach policies: `AmazonS3FullAccess`, `AmazonEC2FullAccess`,
   `AmazonRDSFullAccess`, `IAMReadOnlyAccess`
3. Security credentials → *Create access key* → "Command Line Interface"
4. Copy the **Access Key ID** and **Secret Access Key**
5. Paste them into our chat. I will:
   - Configure a temporary AWS profile locally
   - Run the deployment steps above from this session
   - **Delete the access key immediately after** via `aws iam delete-access-key`

**After deployment, rotate or delete `claude-deployer`** in the IAM console.
Never commit these keys to git.

> Safer alternative: screen-share your AWS console with me and I'll guide
> you click-by-click through the steps above in real time.
