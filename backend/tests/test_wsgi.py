import wsgi


def test_wsgi_module_exposes_flask_app():
    assert wsgi.app.name == "app"
