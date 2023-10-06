from similarity_augmentations.cli import embedding_cmd
from similarity_augmentations.cli.finetune_cmd import app

app.rich_markup_mode = "rich"

app.add_typer(embedding_cmd.app)


if __name__ == "__main__":
    app()
