# Generated by Django 4.0.3 on 2022-05-14 22:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tradeapp', '0016_rename_n_lookup_aimodel_lookup'),
    ]

    operations = [
        migrations.AddField(
            model_name='aimodel',
            name='scaler',
            field=models.FileField(default='hol', upload_to='scalers'),
        ),
    ]
