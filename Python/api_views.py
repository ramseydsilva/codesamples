from django.contrib.auth.models import User
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny
from .serializers import CurrentUserSerializer
from .permissions import IsCurrentUser
from .viewsets import ModelUpdateViewSet


class LoginView(APIView):
    queryset = User.objects.filter(is_active=True)
    permission_classes = (AllowAny,)
    serializer_class = CurrentUserSerializer

    def post(self, request, format=None):
        """
        Login user and receive API token to make authenticated requests
        ---
        parameters:
            - name: username
              description: Username
              type: string
              paramType: form
            - name: password
              description: Password
              type: string
              paramType: form
        response_serializer: CurrentUserSerializer
        """
        username = request.data.get("username")
        password = request.data.get("password")
        remember_me = request.data.get("remember_me")
        if remember_me == "false":
            remember_me = False
        user = authenticate(username=username, password=password)
        if user and user.is_active:
            return Response(login_user(request, user, remember_me=remember_me))
        else:
            raise AuthenticationFailed()

class LogoutView(APIView):
    queryset = User.objects.filter(is_active=True)

    def get(self, request, format=None):
        """
        Logout user and invalidate API token
        """
        if not request.user.is_anonymous():
            token, created = Token.objects.get_or_create(user=request.user)
            token.delete()
        logout(request)
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class GetAuthTokenView(ObtainAuthToken):

    def post(self, *args, **kwargs):
        """
        Get Auth token to make authenticated API requests
        ---
        parameters:
            - name: username
              description: Username
              required: true
            - name: password
              description: Password
              required: true
        response_serializer: rest_framework.authtoken.serializers.AuthTokenSerializer
        responseMessages:
            - code: 400
              message: Incorrect username and/or password
        """
        return super(GetAuthToken, self).post(*args, **kwargs)

class CurrentUserViewSet(ModelUpdateViewSet):
    queryset = User.objects.filter(is_active=True)
    serializer_class = CurrentUserSerializer
    permission_classes = (IsCurrentUser,)
    pagination_class = None

    def get_object(self):
        if self.request.user.is_anonymous():
            user = self.request.user
            user.userprofile = {'contact_number': None, 'address': None, 'wishlist': [], 'product_count': 0, 'store_count': 0}
            return user
        return self.request.user

    def list(self, request, *args, **kwargs):
        """
        Get user profile information"
        """
        serializer = self.get_serializer(self.get_object())
        return Response(serializer.data)
