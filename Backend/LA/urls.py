"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from django.conf import settings
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('hello_world/', views.hello_world, name='hello_world'),
    path('upload/', views.upload_file, name='upload_file'),
    path('plot_dendrogram/', views.plot_dendrogram, name='plot_dendrogram'),
    path('plot_kMeans/', views.plot_KMeans, name='plot_KMeans'),
    path('plot_kMedoids/', views.plot_KMedoids, name='plot_KMedoids'),
    path('plot_BIRCH/', views.plot_BIRCH, name='plot_BIRCH'),
    path('plot_DBSCAN/', views.plot_DBSCAN, name='plot_DBSCAN'),


    # path('get_data/', views.get_data, name='get_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
