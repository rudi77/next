import json
import pika
from typing import Dict
import asyncio

class RabbitMQPublisher:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue_name = "training_jobs"

    async def connect(self):
        # Create a connection to RabbitMQ
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.channel = self.connection.channel()
        
        # Declare the queue
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    async def publish_job(self, job: Dict):
        if not self.channel:
            await self.connect()
            
        # Convert job dict to JSON string
        message = json.dumps(job)
        
        # Publish message
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )

    async def close(self):
        if self.connection:
            self.connection.close() 