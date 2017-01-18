from django.conf.urls import url
from rest_framework_nested import routers as nested_routers
from .views import StoreViewSet, MachineViewSet, DataViewSet, StatusViewSet, MachineModelViewSet, ModelViewSet, LayerViewSet
from .user_views import LoginView, LogoutView, GetAuthTokenView, CurrentUserViewSet
from .routers import UserRouter, DefaultRouter, NestedSimpleRouter

router = DefaultRouter(trailing_slash=True)
router.register(r'stores', StoreViewSet)
router.register(r'models', ModelViewSet)

user_router = UserRouter(trailing_slash=False)
user_router.register(r'user', CurrentUserViewSet)

model_router = NestedSimpleRouter(router, r'models', lookup='model', trailing_slash=False)
model_router.register(r'layers', LayerViewSet)

store_router = NestedSimpleRouter(router, r'stores', lookup='store', trailing_slash=False)
store_router.register(r'machines', MachineViewSet)

machine_router = NestedSimpleRouter(store_router, r'machines', lookup='machine', trailing_slash=False)
machine_router.register(r'statuses', StatusViewSet)
machine_router.register(r'models', MachineModelViewSet)
machine_router.register(r'data', DataViewSet)

urlpatterns = [
    url(r'^login$', LoginView.as_view()),
    url(r'^logout$', LogoutView.as_view()),
    url(r'^api-token-auth$', GetAuthTokenView.as_view())
]

urlpatterns += router.urls
urlpatterns += user_router.urls
urlpatterns += model_router.urls
urlpatterns += store_router.urls
urlpatterns += machine_router.urls
