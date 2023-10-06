import typer

from similarity_augmentations.cli import embedding_cmd

app = typer.Typer(rich_markup_mode="rich")
app.add_typer(embedding_cmd.app)

if __name__ == "__main__":
    app()
