from celery import Celery
from celery.schedules import crontab

# Initialize Celery
# Broker: Redis (default port 6379)
# Backend: Redis (for storing results)
app = Celery('stavki', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

# Configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_always_eager=False, # Set to True for running locally without worker (testing)
)

# Scheduled Tasks (Beat)
app.conf.beat_schedule = {
    'run-orchestrator-hourly': {
        'task': 'src.tasks.run_orchestrator_task',
        'schedule': crontab(minute=0), # Every hour at :00
        # 'schedule': 3600.0, # Alternative: Every 60 minutes
    },
}
