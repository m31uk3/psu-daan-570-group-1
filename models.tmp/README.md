# working directory for saving models

Avoid uploading models saved to git repo as they are large and slow things down.

If you have a model you want to save that is <=50.0 MB move it to either the models.beta or models.stable folders. All other files in models.tmp will be ignored by .gitignore file.
