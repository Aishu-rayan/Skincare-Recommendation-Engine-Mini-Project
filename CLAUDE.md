# See design_plan.md for full design details (ML approach, dataset, AWS architecture, RDS schema, file structure, implementation decisions, dashboard specs).

## Coding conventions
- Python 3.10+
- Use pandas for data manipulation, scipy.sparse for matrix ops
- All S3 paths and RDS credentials via environment variables — 
  never hardcode credentials
- Environment variables needed:
    S3_BUCKET, RDS_HOST, RDS_PORT, RDS_DB, RDS_USER, RDS_PASSWORD
- Use psycopg2 for RDS connections
- Log all job steps with Python logging (INFO level)
- Every script should be runnable locally (reading from data/raw/) AND on SageMaker (reading from S3) via an --env flag:
    --env local | aws
- Use argparse for all CLI scripts

## What to do when asked to build a new feature
1. Check if the relevant src/ module already exists
2. Ensure the --env local | aws pattern is followed
3. No credentials in code — use os.environ.get()
4. Add a brief docstring to every function
5. If writing a SageMaker job, output artifacts to /opt/ml/output/ 
   (SageMaker convention)
