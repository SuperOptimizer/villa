#include "OpsList.hpp"
#include "ui_OpsList.h"

#include <iostream>


OpsList::OpsList(QWidget* parent) : QWidget(parent), ui(new Ui::OpsList)
{
    ui->setupUi(this);
}

OpsList::~OpsList() { delete ui; }

void OpsList::setDataset(z5::Dataset *ds, ChunkCache *cache, float scale)
{
    _ds = ds;
    _cache = cache;
    _scale = scale;
}
