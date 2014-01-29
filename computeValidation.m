function [errors, stepValidation] = computeValidation(layers, params, validation, errors, indexValidation, stepValidation)

nBatchesValidation = length(validation);
errorsTemp = 0;

for k = 1:nBatchesValidation
	layers = fprop(layers, validation(k).data);
	errorsTemp = errorsTemp + mean(computeCost(layers, validation(k).labels, params.cost));
end

errors.validation.cost(indexValidation) = gather(errorsTemp)/nBatchesValidation;
if params.verbose == 1, fprintf('%s: %g	', params.cost, errorsTemp/nBatchesValidation); end
subplot(1,params.nFig,2);
plot((stepValidation(1:indexValidation)), errors.validation.cost(1:indexValidation));
eval(['title(''Validation ', params.cost, ''');']);
drawnow;

if strcmp(params.task, 'class')
	errorsTempClass = 0;
	for k = 1:nBatchesValidation
		layers = fprop(layers, validation(k).data);
		errorsTempClass = errorsTempClass + mean(computeCost(layers, validation(k).labels, 'class'));
    end
    errorsTemp = gather(errorsTempClass);
	errors.validation.class(indexValidation) = errorsTemp / nBatchesValidation;
	subplot(1,params.nFig,3);
	plot((stepValidation(1:indexValidation)), errors.validation.class(1:indexValidation));
	title('Validation classification error');
	drawnow;
	if params.verbose == 1
        fprintf('Class error: %g ', errorsTemp/nBatchesValidation);
    end
end
