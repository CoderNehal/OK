from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Data

@csrf_exempt
def hello_world(request):
    return HttpResponse("Hello World!")

@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        
        # Save the file to the database
        new_data = Data(file=uploaded_file)
        new_data.save()

        return JsonResponse({'message': 'File uploaded successfully'})
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)
    

def get_data(request):
    try:
        first_data = Data.objects.first()
        
        if first_data:
            # Modify this response based on your data model
            response_data = {
                'file_url': first_data.file.url,
                # Add other fields as needed
            }
            return JsonResponse(response_data)
        else:
            return JsonResponse({'error': 'No data available'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)