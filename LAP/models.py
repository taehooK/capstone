from django.db import models
from django.conf import settings
from django.dispatch import receiver
from django.db.models.signals import post_delete


class FileModel(models.Model):
    file = models.FileField(blank=False, null=False)