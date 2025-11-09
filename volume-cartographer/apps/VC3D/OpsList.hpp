#pragma once

#include <QWidget>
#include <ui_OpsList.h>
#include <z5/dataset.hxx>

#include "SurfaceTreeWidget.hpp"
#include "vc/core/util/Slicing.hpp"


class OpsList : public QWidget
{
    Q_OBJECT

public:
    explicit OpsList(QWidget* parent = nullptr);
    ~OpsList();

    void setDataset(z5::Dataset *ds, ChunkCache *cache, float scale);

private:
    Ui::OpsList* ui;

    z5::Dataset *_ds = nullptr;
    ChunkCache *_cache = nullptr;
    float _scale = 0.0;
};

