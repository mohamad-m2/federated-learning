
def test(model, validate_loader, device):
  model.train(False)

  num_data = 0
  correct = 0
  validate = iter(validate_loader)
  for step in range(len(validate)):

    x, y = next(validate)
    num_data += y.size(0)
    x = x.to(device).float()
    y = y.to(device).long()
    output = model(x)
    pred = output.data.max(1)[1]
    correct += pred.eq(y.view(-1)).sum().item()

  return correct/num_data